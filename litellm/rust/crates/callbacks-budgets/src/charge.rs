//! The whole post-call spend counter step for one charged request, I/O included.
//!
//! [`charge_counters`] settles the customer hierarchy counters, names every counter
//! [`plan`](crate::counters::plan) gives once, reads them and the reservation's counters with one
//! batch read, reconciles the reservation, seeds the cold counters, and sends every increment in
//! one pipeline. A counter the reservation holds is left to the reconcile; when the reconcile
//! fails its counters are dropped, read as cold, and charged directly after a reseed. What only the
//! proxy knows comes back through three host calls, each answering a whole batch so the proxy can
//! run its part concurrently: [`BudgetWindows`] (an entity's budget windows from the auth cache or
//! the stored row), [`StoredSpend`] (a counter's spend from the database) and [`CachedSpend`] (the
//! auth cache's spend for a cold counter the database cannot answer).
//!
//! Scopes keep the fan-out's failure rules: a sequential scope drops everything it prepared at its
//! first failure, a concurrent scope keeps every success and each failure separately. All failures
//! are reported after the increments went out and the reservation was stamped.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use futures_util::future::join_all;
use litellm_callbacks::{BudgetReservation, Host, HostCall, HostCallError};
use litellm_redis_store::{
    ErrorKind, Number, RequestRedis, Send,
    spend_counters::{LostSeed, PostCallSettlement, Reading, SpendCounter, SpendCounters},
};
use serde::{Deserialize, Serialize};

use crate::{
    counters::{
        self, CustomerScopeSpend, CustomerWindow, Scope, Task, WindowedEntity, charged,
        entity_label, plan,
    },
    engine::{self, Settle},
    window::{self, Moment},
};

/// One charged request's counter inputs, as the cost callback has them after the spend write.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CounterStep {
    pub token: Option<String>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    pub org_id: Option<String>,
    pub end_user_id: Option<String>,
    pub project_id: Option<String>,
    pub tags: Option<Vec<Option<String>>>,
    pub model_access_groups: Option<Vec<Option<String>>>,
    pub response_cost: Option<f64>,
    /// The entity counters whose committed twin exists for this charge.
    #[serde(deserialize_with = "counters::counter_keys")]
    pub committed_tracked: Vec<SpendCounter>,
    pub customer_windows: Vec<CustomerWindow>,
    /// The customer hierarchy counters the request was admitted under, settled before the rest.
    pub customer_scopes: Vec<CustomerScopeSpend>,
    /// Each reservation entry's `window_reset_at`, in entry order.
    pub window_resets: Vec<Option<String>>,
}

/// Why a counter could not be charged.
#[derive(Debug, thiserror::Error)]
pub enum CounterError {
    #[error(transparent)]
    Store(#[from] litellm_redis_store::Error),
    #[error(transparent)]
    Host(#[from] HostCallError),
}

/// A failure shared by every counter one failed call left unwarmed.
pub type Failure = Arc<CounterError>;

fn store_failure(error: litellm_redis_store::Error) -> Failure {
    Arc::new(error.into())
}

fn host_failure(error: HostCallError) -> Failure {
    Arc::new(error.into())
}

/// What the step did, for the caller to record on its reservation.
#[derive(Debug, Default)]
pub struct CounterOutcome {
    /// The `applied_adjustment` each reservation entry now carries, `None` to leave it as it is.
    pub stamps: Vec<Option<f64>>,
    /// Whether the reservation is settled for good.
    pub finalized: bool,
    /// The post-call pipeline that carried committed twins, which later readers wait on.
    pub committed_settlement: Option<PostCallSettlement>,
    /// The first failure, reported after everything else ran.
    pub error: Option<Failure>,
}

/// One budget window as the proxy resolved it: its duration and the start of the window the
/// request falls in, `None` when the window is not valid.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub(crate) struct BudgetWindow {
    budget_duration: String,
    window_start: Option<String>,
}

/// The entity whose budget windows a [`BudgetWindows`] call reads.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct WindowSource {
    pub(crate) entity_type: WindowedEntity,
    pub(crate) entity_id: String,
    pub(crate) cache_key: String,
}

/// Host call: each entity's budget windows, from the auth cache (`None` when the entity is not
/// cached) or, with `stored`, from its database row.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct BudgetWindows {
    entities: Vec<WindowSource>,
    stored: bool,
}

impl HostCall for BudgetWindows {
    const NAME: &'static str = "budget_windows";
    type Reply = Vec<Option<Vec<BudgetWindow>>>;
}

/// A counter whose spend the database holds: an entity row named by the counter, or a budget
/// window's spend from its maintained row or the spend logs.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "source", rename_all = "snake_case")]
enum StoredCounter {
    Entity {
        counter_key: String,
    },
    Window {
        entity_type: String,
        entity_id: String,
        window_duration: Option<String>,
        window_start: String,
    },
}

/// Host call: each counter's spend from the database, `None` when it has no readable row.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct StoredSpend {
    counters: Vec<StoredCounter>,
}

impl HostCall for StoredSpend {
    const NAME: &'static str = "stored_spend";
    type Reply = Vec<Option<f64>>;
}

/// Host call: for each cold counter, the spend on the first auth cache entry of its sources that
/// is cached, `0.0` when none is.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct CachedSpend {
    sources: Vec<Vec<String>>,
}

impl HostCall for CachedSpend {
    const NAME: &'static str = "cached_spend";
    type Reply = Vec<f64>;
}

/// Whether the database row a counter is named after holds its spend. End user, tag, access group
/// and key, team and user window counters have none.
fn has_stored_spend(counter: &SpendCounter) -> bool {
    matches!(
        counter,
        SpendCounter::Key { window: None, .. }
            | SpendCounter::Team { window: None, .. }
            | SpendCounter::TeamMember { .. }
            | SpendCounter::User(_)
            | SpendCounter::Org(_)
            | SpendCounter::Project(_)
            | SpendCounter::CustomerOrg(_)
            | SpendCounter::CustomerTeam(_)
            | SpendCounter::CustomerMember(_)
            | SpendCounter::CustomerWindow { .. }
    )
}

/// One batch read's answer for a counter: its value, and whether Redis gave it.
#[derive(Clone, Copy, Debug, Default)]
struct Read {
    value: Option<f64>,
    authoritative: bool,
}

type Reads = HashMap<SpendCounter, Read>;

/// Reads every counter with one `MGET`, or one read each when there is no Redis or the `MGET`
/// failed. A value Redis returns is mirrored into memory, as the warm check does.
async fn read_all(store: &SpendCounters, counters: &[SpendCounter]) -> Reads {
    if counters.is_empty() {
        return HashMap::new();
    }
    let batched = match store.read_many(Send::Now, counters) {
        Some(pending) => pending.await.ok().flatten(),
        None => None,
    };
    let reads: Vec<Read> = match batched {
        Some(values) => values
            .into_iter()
            .map(|value| Read {
                value: value.map(Number::as_f64),
                authoritative: true,
            })
            .collect(),
        None => join_all(counters.iter().map(|counter| store.read(counter)))
            .await
            .into_iter()
            .map(|reading| match reading {
                Ok(Reading::Redis(value)) => Read {
                    value: value.map(Number::as_f64),
                    authoritative: true,
                },
                Ok(Reading::Memory(value)) => Read {
                    value: value.map(Number::as_f64),
                    authoritative: false,
                },
                Err(_) => Read::default(),
            })
            .collect(),
    };
    counters
        .iter()
        .zip(reads)
        .map(|(counter, read)| {
            if let (true, Some(value)) = (read.authoritative, read.value)
                && let Err(error) = store.set_local(counter, Number::Float(value), None)
            {
                tracing::debug!(%error, "mirroring a spend counter into memory failed");
            }
            (counter.clone(), read)
        })
        .collect()
}

/// `increment_spend_counter_cache`: a counter's increment on its own, dropped everywhere when it
/// may not have landed.
async fn increment_one(
    store: &SpendCounters,
    counter: &SpendCounter,
    amount: f64,
) -> Result<(), Failure> {
    store
        .increment(counter, Number::Float(amount))
        .await
        .map(drop)
        .map_err(store_failure)
}

fn is_unavailable(error: &litellm_redis_store::Error) -> bool {
    matches!(error.kind(), ErrorKind::BreakerOpen | ErrorKind::Timeout)
}

/// `_apply_spend_counter_increments`: the request's post-call pipeline when it has one, else one
/// pipeline now that drops every counter when it fails. A timed out or breaker-blocked pipeline is
/// dropped silently. Answers with the post-call settlement when it carried a committed twin.
async fn apply(
    store: &Arc<SpendCounters>,
    request: Option<&RequestRedis>,
    pending: &[(SpendCounter, Number)],
) -> Result<Option<PostCallSettlement>, Failure> {
    if pending.is_empty() {
        return Ok(None);
    }
    if let Some(request) = request
        && let Some(settlement) = store.increment_many_post_call(request, pending)
    {
        let committed = pending
            .iter()
            .any(|(counter, _)| matches!(counter, SpendCounter::Committed(_)));
        return Ok(committed.then_some(settlement));
    }
    if let Err(error) = store.increment_many(pending).await {
        if store.has_redis() {
            let counters: Vec<SpendCounter> = pending.iter().map(|(c, _)| c.clone()).collect();
            store.invalidate(&counters).await;
            if is_unavailable(&error) {
                return Ok(None);
            }
        }
        return Err(store_failure(error));
    }
    Ok(None)
}

/// Charges each customer hierarchy counter, seeding the cold ones with their admitted spend first.
async fn settle_customer_spend(
    store: &SpendCounters,
    scopes: &[CustomerScopeSpend],
    cost: f64,
) -> Result<(), Failure> {
    let readings = join_all(scopes.iter().map(|scope| store.read(&scope.counter_key))).await;
    let mut cold = Vec::new();
    for (scope, reading) in scopes.iter().zip(readings) {
        if reading.map_err(store_failure)?.value().is_none() {
            cold.push(scope);
        }
    }
    let seeds = cold.iter().map(|scope| {
        store.seed(
            &scope.counter_key,
            Number::Float(scope.spend),
            LostSeed::Answer,
        )
    });
    for seeded in join_all(seeds).await {
        seeded.map_err(store_failure)?;
    }
    let increments: Vec<(SpendCounter, Number)> = scopes
        .iter()
        .map(|scope| (scope.counter_key.clone(), Number::Float(cost)))
        .collect();
    store
        .increment_many(&increments)
        .await
        .map(drop)
        .map_err(store_failure)
}

/// The `applied_adjustment` every entry with a counter carries once the reserved counters hold the
/// actual cost: [`engine::settled_cost`] less what the entry reserved.
fn settled_stamps(
    reservation: &BudgetReservation,
    step: &CounterStep,
    now: Moment,
) -> Vec<Option<f64>> {
    let actual = step.response_cost.unwrap_or(0.0);
    let default_reserved = reservation.reserved_cost.unwrap_or(0.0);
    reservation
        .entries
        .as_deref()
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let entry = entry.as_ref()?;
            entry.counter_key.as_ref()?;
            let reset = step
                .window_resets
                .get(index)
                .and_then(Option::as_deref)
                .and_then(|reset| window::parse_reset(reset).ok().flatten());
            Some(
                engine::settled_cost(reset, &now, actual)
                    - entry.reserved_cost.unwrap_or(default_reserved),
            )
        })
        .collect()
}

/// The counters a reservation holds, each with the key its entry carries, in entry order.
type Held = Vec<(String, SpendCounter)>;

fn held_counters(reservation: &BudgetReservation) -> Held {
    let mut seen = HashSet::new();
    reservation
        .entries
        .as_deref()
        .unwrap_or_default()
        .iter()
        .flatten()
        .filter_map(|entry| entry.counter_key.as_ref())
        .filter(|key| seen.insert(key.as_str()))
        .filter_map(|key| Some((key.clone(), SpendCounter::parse(key).ok()?)))
        .collect()
}

fn held_counter<'a>(held: &'a Held, key: &str) -> Option<&'a SpendCounter> {
    held.iter()
        .find(|(held_key, _)| held_key == key)
        .map(|(_, counter)| counter)
}

/// What the reconcile left: the counters the reservation still holds, which the charge skips, and
/// the adjustments still to be written. Both are empty when the reconcile failed.
#[derive(Default)]
struct Reconciled {
    reserved: HashSet<SpendCounter>,
    pending: Vec<(SpendCounter, Number)>,
}

/// `reseed_reserved_counter`: raise a counter the reconcile found inconsistent to its stored spend,
/// then charge it the settled cost.
async fn reseed_reserved(
    store: &SpendCounters,
    host: &Host,
    held: &Held,
    reseeds: &[engine::Reseed],
) -> Result<(), Failure> {
    let asked: Vec<(&SpendCounter, f64)> = reseeds
        .iter()
        .filter_map(|reseed| {
            held_counter(held, &reseed.counter_key).map(|counter| (counter, reseed.actual_cost))
        })
        .filter(|(counter, _)| has_stored_spend(counter))
        .collect();
    if asked.is_empty() {
        return Ok(());
    }
    let stored = host
        .call(StoredSpend {
            counters: asked
                .iter()
                .map(|(counter, _)| StoredCounter::Entity {
                    counter_key: counter.key(),
                })
                .collect(),
        })
        .await
        .map_err(host_failure)?;
    for ((counter, actual_cost), spend) in asked.into_iter().zip(stored) {
        let Some(spend) = spend else { continue };
        if let Err(error) = store.set_max(counter, Number::Float(spend)).await {
            tracing::debug!(%error, counter = %counter.key(), "Unable to repair stale spend counter in Redis");
        }
        if actual_cost > 0.0 {
            increment_one(store, counter, actual_cost).await?;
        }
    }
    Ok(())
}

const RECONCILE_FAILED: &str = "Failed to reconcile budget reservation after persisted spend; invalidating reserved counters and falling back to direct increment";

/// `_reconcile_budget_reservation_for_counter_update`: settle the open reservation against the
/// counters it holds, with its adjustments left for the pipeline. On failure the reserved counters
/// are dropped and marked cold in `reads`, so the charge reseeds them before it increments them
/// directly, as the proxy's next warm check would.
async fn reconcile(
    store: &SpendCounters,
    host: &Host,
    reservation: &BudgetReservation,
    held: &Held,
    step: &CounterStep,
    reads: &mut Reads,
    outcome: &mut CounterOutcome,
) -> Reconciled {
    let observed: Vec<Option<f64>> = reservation
        .entries
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|entry| {
            let key = entry.as_ref()?.counter_key.as_deref()?;
            reads.get(held_counter(held, key)?)?.value
        })
        .collect();
    let settled = engine::settle(
        store,
        &Settle {
            reservation,
            window_resets: &step.window_resets,
            observed: &observed,
            actual: step.response_cost.unwrap_or(0.0),
            apply_consistent: false,
        },
        &Moment::now(),
    )
    .await;
    let settlement = match settled {
        Ok(settlement) => match reseed_reserved(store, host, held, &settlement.reseed).await {
            Ok(()) => Some(settlement),
            Err(error) => {
                tracing::warn!(%error, "{RECONCILE_FAILED}");
                None
            }
        },
        Err(error) => {
            tracing::warn!(%error, "{RECONCILE_FAILED}");
            None
        }
    };
    let Some(settlement) = settlement else {
        let dropped: Vec<SpendCounter> = held.iter().map(|(_, counter)| counter.clone()).collect();
        store.invalidate(&dropped).await;
        let cold = Read {
            value: None,
            authoritative: store.has_redis(),
        };
        reads.extend(dropped.into_iter().map(|counter| (counter, cold)));
        return Reconciled::default();
    };
    for (stamp, adjustment) in outcome.stamps.iter_mut().zip(settlement.adjustments) {
        if adjustment.is_some() {
            *stamp = adjustment;
        }
    }
    let pending = settlement
        .pending
        .into_iter()
        .filter_map(|pending| {
            held_counter(held, &pending.counter_key)
                .map(|counter| (counter.clone(), Number::Float(pending.increment)))
        })
        .collect();
    Reconciled {
        reserved: held.iter().map(|(_, counter)| counter.clone()).collect(),
        pending,
    }
}

/// One counter a task warms and charges.
#[derive(Clone, Debug)]
enum Unit {
    /// A lifetime counter, seeded from its row or, without one, from the auth cache.
    Counter {
        counter: SpendCounter,
        sources: Vec<String>,
    },
    /// A budget window, seeded from its window spend. A strict (customer) window is skipped when
    /// only this pod's memory could answer for it.
    Window {
        counter: SpendCounter,
        label: String,
        entity_id: String,
        duration: Option<String>,
        start: String,
        strict: bool,
    },
    /// A committed twin, charged only once it holds the stored spend of the counter it twins.
    Committed {
        base: SpendCounter,
        twin: SpendCounter,
    },
}

impl Unit {
    fn warmed(&self) -> &SpendCounter {
        match self {
            Self::Counter { counter, .. } | Self::Window { counter, .. } => counter,
            Self::Committed { twin, .. } => twin,
        }
    }

    /// A failed window inside an entity's windows is kept beside its siblings; any other failure
    /// fails the task.
    fn fails_task(&self) -> bool {
        !matches!(self, Self::Window { strict: false, .. })
    }
}

/// What one unit came to: charge it, skip it, or a failure.
enum Warmed {
    Charge,
    Skip,
    Failed(Failure),
}

/// A task's units, or the failure that kept it from naming them. `uncached` names an entity whose
/// windows wait for its stored row.
#[derive(Default)]
struct TaskUnits {
    units: Vec<usize>,
    uncached: Vec<WindowSource>,
    failure: Option<Failure>,
}

/// Every unit a set of scopes names, and each scope's tasks as indexes into them.
#[derive(Default)]
struct Named {
    units: Vec<Unit>,
    scopes: Vec<(bool, Vec<TaskUnits>)>,
}

impl Named {
    fn counters(&self) -> Vec<SpendCounter> {
        self.units
            .iter()
            .map(|unit| unit.warmed().clone())
            .collect()
    }
}

/// Warms every unit: one batch read, one stored spend call for the cold ones, their seeds run
/// together, then one cached spend call for the counters the database could not answer. A unit on a
/// counter the reservation holds is skipped.
async fn warm(
    store: &SpendCounters,
    host: &Host,
    units: &[Unit],
    reads: &Reads,
    reserved: &HashSet<SpendCounter>,
) -> Vec<Warmed> {
    let mut warmed: Vec<Option<Warmed>> = units
        .iter()
        .map(|unit| {
            let counter = unit.warmed();
            if reserved.contains(counter) {
                return Some(Warmed::Skip);
            }
            reads
                .get(counter)
                .and_then(|read| read.value)
                .map(|_| Warmed::Charge)
        })
        .collect();
    let mut asked = Vec::new();
    let mut fallback = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        if warmed[index].is_some() {
            continue;
        }
        let stored = match unit {
            Unit::Counter { counter, .. } if has_stored_spend(counter) => {
                Some(StoredCounter::Entity {
                    counter_key: counter.key(),
                })
            }
            Unit::Counter { .. } => {
                fallback.push(index);
                None
            }
            Unit::Window {
                counter,
                label,
                entity_id,
                duration,
                start,
                strict,
            } => {
                let authoritative = reads.get(counter).is_some_and(|read| read.authoritative);
                if *strict && store.has_redis() && !authoritative {
                    tracing::warn!(
                        counter = counter.key(),
                        "Skipping cold spend counter seed because window spend could not be loaded"
                    );
                    warmed[index] = Some(Warmed::Skip);
                    None
                } else {
                    Some(StoredCounter::Window {
                        entity_type: label.clone(),
                        entity_id: entity_id.clone(),
                        window_duration: duration.clone(),
                        window_start: start.clone(),
                    })
                }
            }
            Unit::Committed { base, .. } => Some(StoredCounter::Entity {
                counter_key: base.key(),
            }),
        };
        if let Some(stored) = stored {
            asked.push((index, stored));
        }
    }
    if !asked.is_empty() {
        let request = StoredSpend {
            counters: asked.iter().map(|(_, stored)| stored.clone()).collect(),
        };
        match host.call(request).await {
            Ok(spends) => {
                let seeds = asked
                    .iter()
                    .zip(spends.into_iter().chain(std::iter::repeat(None)))
                    .map(|((index, _), spend)| seed(store, &units[*index], spend));
                let seeded = join_all(seeds).await;
                for ((index, _), seeded) in asked.iter().zip(seeded) {
                    match seeded {
                        Seeded::Done(result) => warmed[*index] = Some(result),
                        Seeded::Fallback => fallback.push(*index),
                    }
                }
            }
            Err(error) => {
                let failure = host_failure(error);
                for (index, _) in &asked {
                    warmed[*index] = Some(Warmed::Failed(failure.clone()));
                }
            }
        }
    }
    if !fallback.is_empty() {
        fallback.sort_unstable();
        let sources = fallback
            .iter()
            .map(|index| match &units[*index] {
                Unit::Counter { sources, .. } => sources.clone(),
                _ => Vec::new(),
            })
            .collect();
        match host.call(CachedSpend { sources }).await {
            Ok(spends) => {
                let increments = fallback
                    .iter()
                    .zip(spends.into_iter().chain(std::iter::repeat(0.0)))
                    .map(|(index, spend)| async move {
                        let counter = units[*index].warmed();
                        if spend > 0.0 {
                            increment_one(store, counter, spend).await?;
                        }
                        Ok::<(), Failure>(())
                    });
                for (index, result) in fallback.iter().zip(join_all(increments).await) {
                    warmed[*index] = Some(match result {
                        Ok(()) => Warmed::Charge,
                        Err(failure) => Warmed::Failed(failure),
                    });
                }
            }
            Err(error) => {
                let failure = host_failure(error);
                for index in &fallback {
                    warmed[*index] = Some(Warmed::Failed(failure.clone()));
                }
            }
        }
    }
    warmed
        .into_iter()
        .map(|warmed| warmed.unwrap_or(Warmed::Skip))
        .collect()
}

enum Seeded {
    Done(Warmed),
    Fallback,
}

/// `SpendCounterReseed.coalesced(_window)` and `seed_committed_spend` past their DB read: seed the
/// cold counter with `SET NX` so one pod wins.
async fn seed(store: &SpendCounters, unit: &Unit, spend: Option<f64>) -> Seeded {
    match (unit, spend) {
        (Unit::Counter { .. }, None) => Seeded::Fallback,
        (Unit::Counter { counter, .. }, Some(spend)) => Seeded::Done(
            match store
                .seed(counter, Number::Float(spend), LostSeed::Answer)
                .await
            {
                Ok(_) => Warmed::Charge,
                Err(error) => Warmed::Failed(store_failure(error)),
            },
        ),
        (Unit::Window { counter, .. }, None) => {
            tracing::warn!(
                counter = counter.key(),
                "Skipping cold spend counter seed because window spend could not be loaded"
            );
            Seeded::Done(Warmed::Skip)
        }
        (
            Unit::Window {
                counter, strict, ..
            },
            Some(spend),
        ) => Seeded::Done(
            match store
                .seed(counter, Number::Float(spend), LostSeed::Increment)
                .await
            {
                Ok(_) => Warmed::Charge,
                Err(_) if *strict => Warmed::Skip,
                Err(error) => Warmed::Failed(store_failure(error)),
            },
        ),
        (Unit::Committed { .. }, None) => Seeded::Done(Warmed::Skip),
        (Unit::Committed { twin, .. }, Some(spend)) => Seeded::Done(
            match store
                .seed(twin, Number::Float(spend), LostSeed::Answer)
                .await
            {
                Ok(_) => Warmed::Charge,
                Err(error) => {
                    tracing::debug!(%error, twin = twin.key(), "committed spend seed failed");
                    Warmed::Skip
                }
            },
        ),
    }
}

/// Each entity's budget windows by its auth cache key, or the failure that kept them from being
/// read.
type Windows = HashMap<String, Result<Option<Vec<BudgetWindow>>, Failure>>;

/// The step's state while it turns the plan into increments.
struct Run<'a> {
    store: &'a Arc<SpendCounters>,
    host: &'a Host,
    step: &'a CounterStep,
    cost: f64,
    /// The counters the open reservation holds, whose invalid windows are not worth a warning.
    held: HashSet<SpendCounter>,
}

/// What a set of scopes came to: the increments, the entities whose windows wait for their stored
/// row, and every failure.
#[derive(Default)]
struct Charged {
    pending: Vec<(SpendCounter, Number)>,
    uncached: Vec<WindowSource>,
    failures: Vec<Failure>,
}

impl Run<'_> {
    fn invalid_window(&self, counter: &SpendCounter) {
        if !self.held.contains(counter) {
            tracing::warn!(
                counter = counter.key(),
                "Skipping spend counter increment for invalid budget window"
            );
        }
    }

    /// The units of one task, appended to `units`.
    fn task_units(&self, task: &Task, windows: &Windows, units: &mut Vec<Unit>) -> TaskUnits {
        let mut named = TaskUnits::default();
        let mut push = |unit: Unit| {
            named.units.push(units.len());
            units.push(unit);
        };
        match task {
            Task::Counter { counter, sources } => push(Unit::Counter {
                counter: counter.clone(),
                sources: sources.clone(),
            }),
            Task::CachedWindows(source) => match windows.get(&source.cache_key) {
                Some(Ok(Some(windows))) => {
                    for window in windows {
                        let counter = source
                            .entity_type
                            .window(&source.entity_id, &window.budget_duration);
                        match &window.window_start {
                            Some(start) => push(Unit::Window {
                                counter,
                                label: <&str>::from(source.entity_type).to_owned(),
                                entity_id: source.entity_id.clone(),
                                duration: Some(window.budget_duration.clone()),
                                start: start.clone(),
                                strict: false,
                            }),
                            None => self.invalid_window(&counter),
                        }
                    }
                }
                Some(Ok(None)) | None => named.uncached.push(source.clone()),
                Some(Err(failure)) => named.failure = Some(failure.clone()),
            },
            Task::CustomerWindow(index) => {
                let window = &self.step.customer_windows[*index];
                match &window.window_start {
                    Some(start) => {
                        let label = entity_label(&window.entity_type);
                        push(Unit::Window {
                            counter: window.counter_key.clone(),
                            strict: label.starts_with("customer_"),
                            label,
                            entity_id: window.entity_id.clone(),
                            duration: window.window_duration.clone(),
                            start: start.clone(),
                        });
                    }
                    None => self.invalid_window(&window.counter_key),
                }
            }
            Task::Committed(twins) => {
                for (base, twin) in twins {
                    push(Unit::Committed {
                        base: base.clone(),
                        twin: twin.clone(),
                    });
                }
            }
        }
        named
    }

    /// Names every unit of the scopes, once.
    fn name(&self, scopes: &[Scope], windows: &Windows) -> Named {
        let mut units = Vec::new();
        let scopes = scopes
            .iter()
            .map(|scope| {
                let tasks = scope
                    .tasks
                    .iter()
                    .map(|task| self.task_units(task, windows, &mut units))
                    .collect();
                (scope.concurrent, tasks)
            })
            .collect();
        Named { units, scopes }
    }

    /// Reads the auth cache's budget windows for every entity that has them.
    async fn cached_windows(&self, sources: Vec<WindowSource>, stored: bool) -> Windows {
        if sources.is_empty() {
            return HashMap::new();
        }
        let keys: Vec<String> = sources.iter().map(|s| s.cache_key.clone()).collect();
        match self
            .host
            .call(BudgetWindows {
                entities: sources,
                stored,
            })
            .await
        {
            Ok(windows) => keys
                .into_iter()
                .zip(windows.into_iter().chain(std::iter::repeat(None)))
                .map(|(key, windows)| (key, Ok(windows)))
                .collect(),
            Err(error) => {
                let failure = host_failure(error);
                keys.into_iter()
                    .map(|key| (key, Err(failure.clone())))
                    .collect()
            }
        }
    }

    /// Charges the named scopes: warms their units, skipping the reserved ones, and folds each
    /// scope's results by its failure rule.
    async fn charge(
        &self,
        named: Named,
        reads: &Reads,
        reserved: &HashSet<SpendCounter>,
    ) -> Charged {
        let Named { units, scopes } = named;
        let warmed = warm(self.store, self.host, &units, reads, reserved).await;
        let mut charged = Charged::default();
        for (concurrent, tasks) in scopes {
            let mut scope = Charged::default();
            let mut scope_failure = None;
            for task in tasks {
                let mut task_failure = task.failure;
                let mut task_pending = Vec::new();
                for index in task.units {
                    match &warmed[index] {
                        Warmed::Charge => task_pending
                            .push((units[index].warmed().clone(), Number::Float(self.cost))),
                        Warmed::Skip => {}
                        Warmed::Failed(failure) if units[index].fails_task() => {
                            task_failure.get_or_insert_with(|| failure.clone());
                        }
                        Warmed::Failed(failure) => scope.failures.push(failure.clone()),
                    }
                }
                match task_failure {
                    Some(failure) if concurrent => scope.failures.push(failure),
                    Some(failure) => {
                        scope_failure = Some(failure);
                        break;
                    }
                    None => {
                        scope.pending.extend(task_pending);
                        scope.uncached.extend(task.uncached);
                    }
                }
            }
            if let Some(failure) = scope_failure {
                charged.failures.push(failure);
                continue;
            }
            charged.pending.extend(scope.pending);
            charged.uncached.extend(scope.uncached);
            charged.failures.extend(scope.failures);
        }
        charged
    }
}

/// Runs the post-call counter step. Every store write that can run together does, and the proxy is
/// asked for what only it knows through `host`. The increments ride the request's post-call
/// pipeline when `request` is given and there is Redis.
pub async fn charge_counters(
    store: &Arc<SpendCounters>,
    host: &Host,
    request: Option<&RequestRedis>,
    step: &CounterStep,
    reservation: Option<&BudgetReservation>,
) -> CounterOutcome {
    let entries = reservation
        .and_then(|reservation| reservation.entries.as_ref())
        .map_or(0, Vec::len);
    let mut outcome = CounterOutcome {
        stamps: vec![None; entries],
        ..CounterOutcome::default()
    };
    let cost = charged(step.response_cost);
    if let Some(cost) = cost
        && !step.customer_scopes.is_empty()
        && let Err(failure) = settle_customer_spend(store, &step.customer_scopes, cost).await
    {
        outcome.error = Some(failure);
        return outcome;
    }
    let open = reservation.filter(|reservation| reservation.finalized != Some(true));
    let held = open.map(held_counters).unwrap_or_default();
    let run = Run {
        store,
        host,
        step,
        cost: cost.unwrap_or(0.0),
        held: held.iter().map(|(_, counter)| counter.clone()).collect(),
    };
    let named = match cost {
        Some(_) => {
            let scopes = plan(step);
            let sources = scopes
                .iter()
                .flat_map(|scope| &scope.tasks)
                .filter_map(|task| match task {
                    Task::CachedWindows(source) => Some(source.clone()),
                    _ => None,
                })
                .collect();
            let windows = run.cached_windows(sources, false).await;
            run.name(&scopes, &windows)
        }
        None => Named::default(),
    };
    let mut counters: Vec<SpendCounter> = held.iter().map(|(_, counter)| counter.clone()).collect();
    counters.extend(named.counters());
    let mut seen = HashSet::new();
    counters.retain(|counter| seen.insert(counter.clone()));
    let mut reads = read_all(store, &counters).await;
    let reconciled = match open {
        Some(open) => reconcile(store, host, open, &held, step, &mut reads, &mut outcome).await,
        None => Reconciled::default(),
    };
    if cost.is_none() {
        if let Err(failure) = apply(store, request, &reconciled.pending).await {
            outcome.error = Some(failure);
            return outcome;
        }
        return finish(outcome, reservation, step, Vec::new());
    }
    let charged = run.charge(named, &reads, &reconciled.reserved).await;
    let mut pending = reconciled.pending;
    pending.extend(charged.pending);
    match apply(store, request, &pending).await {
        Ok(settlement) => outcome.committed_settlement = settlement,
        Err(failure) => {
            outcome.error = Some(failure);
            return outcome;
        }
    }
    let mut failures = charged.failures;
    if !charged.uncached.is_empty() {
        let stored_windows: Windows = run
            .cached_windows(charged.uncached.clone(), true)
            .await
            .into_iter()
            .map(|(key, windows)| {
                (
                    key,
                    windows.map(|windows| Some(windows.unwrap_or_default())),
                )
            })
            .collect();
        let deferred = run.name(
            &[Scope::concurrent(
                charged.uncached.into_iter().map(Task::CachedWindows),
            )],
            &stored_windows,
        );
        let reads = read_all(store, &deferred.counters()).await;
        let later = run.charge(deferred, &reads, &reconciled.reserved).await;
        failures.extend(later.failures);
        if let Err(failure) = apply(store, request, &later.pending).await {
            outcome.error = Some(failure);
            return outcome;
        }
    }
    finish(outcome, reservation, step, failures)
}

fn finish(
    mut outcome: CounterOutcome,
    reservation: Option<&BudgetReservation>,
    step: &CounterStep,
    failures: Vec<Failure>,
) -> CounterOutcome {
    if let Some(reservation) = reservation {
        outcome.stamps = settled_stamps(reservation, step, Moment::now());
    }
    match failures.into_iter().next() {
        Some(failure) => outcome.error = Some(failure),
        None => outcome.finalized = reservation.is_some(),
    }
    outcome
}
