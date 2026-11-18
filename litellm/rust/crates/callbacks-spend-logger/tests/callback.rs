//! Drives the spend log callback through its hooks with a scripted host, over a writer whose
//! ledger store records what it is asked to insert. Row inputs come from the row builder's
//! goldens, which `callbacks-spend-log/scripts/record_goldens.py` recorded from Python.

use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use litellm_callbacks::{
    BudgetReservation, Callback, Field, FieldSet, FieldValues, Hook, HookInput, HostRequest,
    PatchOp, ReservationEntry, Verdict, hosted,
};
use litellm_callbacks_budgets::Budgets;
use litellm_callbacks_spend_logger::{SpendLogCallback, SpendLogParts};
use litellm_coroutine::CoroutineState;
use litellm_redis_store::{
    Number,
    spend_counters::{SpendCounter, SpendCounters},
};
use litellm_repositories::SpendLogWrite;
use litellm_spend_log_writer::{
    LedgerStore, QueuedSpendLog, SpendLogFlusher, SpendLogWriter, WriterSettings,
};
use litellm_tracing::TokenRedactor;
use rstest::{fixture, rstest};
use serde_json::{Value as Json, json};

const KEY_COUNTER: &str = "spend:key:hashed";
const REQUEST_ID: &str = "chatcmpl-golden";

#[derive(Clone, Default)]
struct Store {
    written: Arc<Mutex<Vec<SpendLogWrite>>>,
    accepts: Option<usize>,
}

impl LedgerStore for Store {
    async fn insert(&self, writes: &[SpendLogWrite]) -> Result<usize, litellm_db::Error> {
        self.written.lock().unwrap().extend_from_slice(writes);
        Ok(self.accepts.unwrap_or(writes.len()))
    }
}

struct Harness {
    callback: Arc<SpendLogCallback<Store>>,
    counters: Arc<SpendCounters>,
    store: Store,
    wakes: Arc<AtomicUsize>,
}

fn harness(store: Store) -> Harness {
    let counters = Arc::new(SpendCounters::new(None));
    counters
        .set_local(
            &SpendCounter::parse(KEY_COUNTER).unwrap(),
            Number::Float(5.0),
            None,
        )
        .unwrap();
    let wakes = Arc::new(AtomicUsize::new(0));
    let woken = Arc::clone(&wakes);
    let writer = Arc::new(SpendLogWriter::new(
        SpendLogFlusher::new(store.clone(), WriterSettings::default()),
        move || {
            woken.fetch_add(1, Ordering::Relaxed);
        },
    ));
    Harness {
        callback: Arc::new(SpendLogCallback::new(SpendLogParts {
            writer,
            budgets: Arc::new(Budgets::new(Arc::clone(&counters))),
            tokens: Arc::new(TokenRedactor::new(16, Vec::new())),
            retention: None,
        })),
        counters,
        store,
        wakes,
    }
}

/// A recorded success row input for a completion, with its config, minus the request id the
/// callback takes from `spend_log_id`.
#[fixture]
fn golden() -> (Json, Json) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../callbacks-spend-log/tests/goldens/rows.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let mut record: Json = text
        .lines()
        .filter_map(|line| serde_json::from_str::<Json>(&line.replace("#~#", "")).ok())
        .find(|record| {
            record["input"]["call_type"] == "acompletion"
                && record["input"]["standard_logging_object"].is_object()
                && record["input"]["kwargs"]["model"].is_string()
                && record["output"]
                    .as_str()
                    .is_some_and(|output| output.contains("\"status\": \"success\""))
                && record["config"]["turn_off_message_logging"] == false
        })
        .unwrap();
    record["input"]
        .as_object_mut()
        .unwrap()
        .remove("request_id");
    (record["config"].clone(), record["input"].clone())
}

fn settings(config: &Json) -> Json {
    json!({
        "disable_spend_updates": false,
        "disable_spend_logs": false,
        "disable_error_logs": false,
        "disable_responses_id_security": false,
        "row_config": config,
        "proxy_budget_name": "litellm-proxy-budget",
    })
}

fn identity() -> Json {
    json!({"token": "hashed", "user_id": "user-1", "team_id": "team-1", "end_user_id": null, "key_alias": "alias"})
}

fn reservation() -> BudgetReservation {
    BudgetReservation {
        reserved_cost: Some(2.0),
        entries: Some(vec![Some(ReservationEntry {
            counter_key: Some(KEY_COUNTER.to_owned()),
            entity_type: Some("key".to_owned()),
            entity_id: Some("hashed".to_owned()),
            reserved_cost: Some(2.0),
            applied_adjustment: Some(0.0),
            ..ReservationEntry::default()
        })]),
        ..BudgetReservation::default()
    }
}

fn reads() -> FieldSet {
    FieldSet::of(&[
        Field::SpendLogInput,
        Field::SpendLogId,
        Field::Cost,
        Field::BilledRetryAttempts,
        Field::BudgetReservation,
        Field::Outcome,
    ])
}

fn input(
    hook: Hook,
    spend_log_input: Json,
    cost: Json,
    reservation: Option<BudgetReservation>,
) -> HookInput {
    let mut fields = FieldValues::new(reads())
        .with(Field::SpendLogInput, spend_log_input)
        .with(Field::SpendLogId, json!({"id": REQUEST_ID}))
        .with(Field::Cost, cost);
    if let Some(reservation) = reservation {
        fields.insert(
            Field::BudgetReservation,
            serde_json::to_value(reservation).unwrap(),
        );
    }
    HookInput {
        hook,
        fields,
        call: None,
    }
}

fn priced(cost: f64) -> Json {
    json!({"response_cost": cost})
}

/// One `null` per batch item, the proxy's answer when nothing is cached or stored.
fn nulls(request: &Json, items: &str) -> Json {
    Json::Array(vec![
        Json::Null;
        request[items].as_array().map_or(0, Vec::len)
    ])
}

struct Ran {
    calls: Vec<(String, Json)>,
    written_back: Option<BudgetReservation>,
}

/// Runs one hook with a host that answers every call and records them.
async fn run(harness: &Harness, input: HookInput, chains: bool) -> Ran {
    let callback = Arc::clone(&harness.callback);
    let mut body = hosted(|host| async move { callback.handle_hosted(&input, None, host).await });
    let mut calls = Vec::new();
    loop {
        match body.resume().await.unwrap() {
            CoroutineState::Complete(output) => {
                let written_back = match output.unwrap() {
                    Some(Verdict::Patch(patch)) => match &patch.ops[..] {
                        [
                            PatchOp::Set {
                                field: Field::BudgetReservation,
                                value,
                            },
                        ] => Some(serde_json::from_value(value.clone()).unwrap()),
                        other => panic!("unexpected patch {other:?}"),
                    },
                    None => None,
                    Some(other) => panic!("unexpected verdict {other:?}"),
                };
                return Ran {
                    calls,
                    written_back,
                };
            }
            CoroutineState::Yielded(HostRequest {
                call,
                request,
                reply,
            }) => {
                let answer = match call {
                    "accounting_context" => json!({
                        "windows": [],
                        "savings": request["built"].is_object().then(|| json!({"compression": 0.0, "prompt_caching": 0.0})),
                        "model_access_groups": [],
                        "customer_windows": [],
                        "committed_tracked": [],
                        "customer_entities": [],
                        "chains_through_db": chains,
                    }),
                    "increment_search_free_tier" => json!(SEARCH_COUNTER_TOTAL_AFTER),
                    "enrich_key_identity" => {
                        json!({"user_id": "enriched-user", "team_alias": "enriched-team"})
                    }
                    "budget_windows" => nulls(&request, "entities"),
                    "stored_spend" => nulls(&request, "counters"),
                    "cached_spend" => {
                        Json::Array(vec![
                            json!(0.0);
                            request["sources"].as_array().map_or(0, Vec::len)
                        ])
                    }
                    _ => Json::Null,
                };
                calls.push((call.to_owned(), request));
                reply.send(Ok(answer));
            }
        }
    }
}

const SEARCH_COUNTER_TOTAL_AFTER: f64 = 6.0;

fn success(config: &Json, row: Json) -> Json {
    json!({
        "kind": "success",
        "settings": settings(config),
        "identity": identity(),
        "call_type": "acompletion",
        "row": {"row": row},
    })
}

fn queued(harness: &Harness) -> Vec<QueuedSpendLog> {
    harness.callback.writer().take(100)
}

fn row_json(entry: &QueuedSpendLog) -> Json {
    serde_json::from_str(&entry.to_json()).unwrap()
}

fn counter(harness: &Harness, key: &str) -> Option<f64> {
    harness
        .counters
        .get_local(&SpendCounter::parse(key).unwrap())
        .unwrap()
        .map(Number::as_f64)
}

fn names(calls: &[(String, Json)]) -> Vec<&str> {
    calls.iter().map(|(name, _)| name.as_str()).collect()
}

fn call<'a>(calls: &'a [(String, Json)], name: &str) -> &'a Json {
    &calls
        .iter()
        .find(|(called, _)| called == name)
        .unwrap_or_else(|| panic!("{name} was not called"))
        .1
}

#[rstest]
#[tokio::test]
async fn a_priced_success_queues_one_ledger_row_under_the_typed_request_id(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());

    let ran = run(
        &harness,
        input(
            Hook::AttemptEnded,
            success(&config, row),
            priced(0.25),
            None,
        ),
        false,
    )
    .await;

    assert_eq!(
        names(&ran.calls),
        [
            "accounting_context",
            "after_write",
            "budget_windows",
            "stored_spend",
            "cached_spend",
            "budget_windows",
            "after_charge",
        ]
    );
    assert_eq!(counter(&harness, "spend:user:user-1"), Some(0.25));
    let rows = queued(&harness);
    assert_eq!(rows.len(), 1);
    let stored = row_json(&rows[0]);
    assert_eq!(stored["request_id"], json!(REQUEST_ID));
    assert_eq!(stored["spend"], json!(0.25));
    assert_eq!(stored["accounting_log_disabled"], json!(false));
    assert!(stored["accounting_payload"].is_object());
    assert!(stored["accounting_delivery_id"].is_string());
    let after_write = call(&ran.calls, "after_write");
    assert_eq!(after_write["row"]["spend"], json!(0.25));
    assert_eq!(after_write["claimed"], json!(false));
    assert_eq!(after_write["spend_logs_kept"], json!(true));
    assert_eq!(
        call(&ran.calls, "after_charge")["response_cost"],
        json!(0.25)
    );
    assert_eq!(harness.wakes.load(Ordering::Relaxed), 0);
}

#[rstest]
#[tokio::test]
async fn spend_logs_off_queue_only_the_charge(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["settings"]["disable_spend_logs"] = json!(true);

    run(
        &harness,
        input(Hook::AttemptEnded, request, priced(0.25), None),
        false,
    )
    .await;

    let stored = row_json(&queued(&harness)[0]);
    assert_eq!(stored["accounting_log_disabled"], json!(true));
    assert!(stored.get("metadata").is_none());
}

#[rstest]
#[tokio::test]
async fn a_charged_reservation_is_settled_and_written_back(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());

    let ran = run(
        &harness,
        input(
            Hook::AttemptEnded,
            success(&config, row),
            priced(0.5),
            Some(reservation()),
        ),
        false,
    )
    .await;

    assert_eq!(counter(&harness, KEY_COUNTER), Some(3.5));
    let settled = ran
        .written_back
        .expect("the settled reservation is written back");
    assert_eq!(settled.finalized, Some(true));
    let entry = settled.entries.unwrap()[0].clone().unwrap();
    assert_eq!(entry.applied_adjustment, Some(-1.5));
}

#[rstest]
#[tokio::test]
async fn disabled_spend_updates_skip_the_row_and_release_the_reservation(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["settings"]["disable_spend_updates"] = json!(true);

    let ran = run(
        &harness,
        input(
            Hook::AttemptEnded,
            request,
            priced(0.25),
            Some(reservation()),
        ),
        false,
    )
    .await;

    assert!(ran.calls.is_empty());
    assert!(queued(&harness).is_empty());
    assert_eq!(counter(&harness, KEY_COUNTER), Some(3.0));
    assert_eq!(ran.written_back.unwrap().finalized, Some(true));
}

#[rstest]
#[case::deferred_batch(priced(0.25), json!({"defers_batch_cost": true}), Some(3.0))]
#[case::unbilled_interaction(json!({}), json!({"unbilled_interaction": true}), Some(3.0))]
#[case::interaction_still_running(json!({}), json!({"unbilled_interaction": true, "interaction_in_progress": true}), Some(5.0))]
#[case::unpriced(json!({}), json!({}), Some(3.0))]
#[tokio::test]
async fn an_uncharged_success_releases_or_keeps_the_reservation(
    golden: (Json, Json),
    #[case] cost: Json,
    #[case] response: Json,
    #[case] counter_after: Option<f64>,
) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["response"] = response;
    request["settings"]["background_interaction_cost_polling"] = json!(true);

    run(
        &harness,
        input(Hook::AttemptEnded, request, cost, Some(reservation())),
        false,
    )
    .await;

    assert!(queued(&harness).is_empty());
    assert_eq!(counter(&harness, KEY_COUNTER), counter_after);
}

#[rstest]
#[tokio::test]
async fn an_unpriced_success_raises_the_failed_tracking_alert_and_fails_the_batch(
    golden: (Json, Json),
) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row.clone());
    request["batch_accounting"] = json!(true);

    let ran = run(
        &harness,
        input(
            Hook::AttemptEnded,
            request,
            json!({"response_cost_failure_debug_info": "no price for the model"}),
            Some(reservation()),
        ),
        false,
    )
    .await;

    assert_eq!(
        names(&ran.calls),
        ["failed_tracking_alert", "batch_accounting"]
    );
    let alert = &ran.calls[0].1;
    let model = row["kwargs"]["model"].as_str().unwrap();
    assert_eq!(alert["failing_model"], json!(model));
    let message = alert["error_message"].as_str().unwrap();
    assert!(message.contains("no price for the model"), "{message}");
    assert!(message.contains("call_type: acompletion"), "{message}");
    assert_eq!(
        ran.calls[1].1,
        json!({"write_started": true, "failed": true})
    );
}

#[rstest]
#[tokio::test]
async fn a_charged_success_without_a_reservation_starts_the_batch_write(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["batch_accounting"] = json!(true);

    let ran = run(
        &harness,
        input(Hook::AttemptEnded, request, priced(0.25), None),
        false,
    )
    .await;

    assert_eq!(
        *call(&ran.calls, "batch_accounting"),
        json!({"write_started": true, "failed": false})
    );
}

fn grounded(row: &Json, provider: &str) -> Json {
    let mut row = row.clone();
    row["usage"]["prompt_tokens_details"] = json!({"web_search_requests": 3});
    row["standard_logging_object"]["custom_llm_provider"] = json!(provider);
    row["standard_logging_object"]["cost_breakdown"] = json!({"input_cost": 0.5});
    row["standard_logging_object"]["model_map_information"] = json!({"model_map_value": {
        "litellm_provider": provider,
        "web_search_billing_unit": "per_query",
        "search_context_cost_per_query": {"search_context_size_medium": 0.1},
        "search_free_requests": {"count": 5, "period": "day", "shared_group": "g3"},
    }});
    row
}

#[rstest]
#[tokio::test]
async fn grounding_under_the_free_allowance_is_not_charged(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, grounded(&row, "gemini"));
    request["settings"]["apply_search_free_tier"] = json!(true);

    let ran = run(
        &harness,
        input(Hook::AttemptEnded, request, priced(1.0), None),
        false,
    )
    .await;

    let counted = call(&ran.calls, "increment_search_free_tier");
    let day = counted["key"]
        .as_str()
        .unwrap()
        .strip_prefix("litellm:search_free_tier:gemini:g3:")
        .unwrap();
    assert!(
        chrono::NaiveDate::parse_from_str(day, "%Y-%m-%d").is_ok(),
        "{day}"
    );
    assert_eq!(counted["units"], json!(3));
    let units = 3.0;
    let freed = (5.0 - (SEARCH_COUNTER_TOTAL_AFTER - units)).min(units);
    let charged = 1.0 - 0.1 * units * freed / units;
    let user = counter(&harness, "spend:user:user-1").unwrap();
    assert!((user - charged).abs() < 1e-12, "{user} != {charged}");
    assert_eq!(row_json(&queued(&harness)[0])["spend"], json!(charged));
    assert_eq!(
        call(&ran.calls, "after_charge")["response_cost"],
        json!(charged)
    );
}

#[rstest]
#[tokio::test]
async fn a_finalized_cost_does_not_draw_from_the_allowance_again(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, grounded(&row, "gemini"));
    request["settings"]["apply_search_free_tier"] = json!(true);

    let ran = run(
        &harness,
        input(
            Hook::AttemptEnded,
            request,
            json!({"response_cost": 1.0, "cost_finalized": true}),
            None,
        ),
        false,
    )
    .await;

    assert!(!names(&ran.calls).contains(&"increment_search_free_tier"));
    assert_eq!(row_json(&queued(&harness)[0])["spend"], json!(1.0));
}

#[rstest]
#[tokio::test]
async fn a_responses_turn_the_next_turn_reads_back_requests_a_flush(golden: (Json, Json)) {
    let (config, mut row) = golden;
    row["call_type"] = json!("aresponses");
    let harness = harness(Store::default());

    run(
        &harness,
        input(
            Hook::AttemptEnded,
            success(&config, row),
            priced(0.25),
            None,
        ),
        true,
    )
    .await;

    assert_eq!(harness.wakes.load(Ordering::Relaxed), 1);
    assert_eq!(queued(&harness).len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_request_with_no_identity_is_not_charged(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["identity"] = json!({});

    let ran = run(
        &harness,
        input(Hook::AttemptEnded, request, priced(0.25), None),
        false,
    )
    .await;

    assert!(ran.calls.is_empty());
    assert!(queued(&harness).is_empty());
}

#[rstest]
#[tokio::test]
async fn a_failed_attempt_is_left_to_the_fallback(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut hook = input(
        Hook::AttemptEnded,
        success(&config, row),
        priced(0.25),
        None,
    );
    hook.fields.insert(Field::Outcome, json!("failure"));

    run(&harness, hook, false).await;

    assert!(queued(&harness).is_empty());
}

#[rstest]
#[tokio::test]
async fn enrichment_fills_the_row_metadata_the_key_left_empty(golden: (Json, Json)) {
    let (config, mut row) = golden;
    row["metadata"]["user_api_key_user_id"] = Json::Null;
    let harness = harness(Store::default());
    let mut request = success(&config, row);
    request["identity"]["user_id"] = Json::Null;

    let ran = run(
        &harness,
        input(Hook::AttemptEnded, request, priced(0.25), None),
        false,
    )
    .await;

    assert_eq!(names(&ran.calls)[0], "enrich_key_identity");
    assert_eq!(ran.calls[1].1["user_id"], json!("enriched-user"));
    let metadata = row_json(&queued(&harness)[0])["metadata"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(metadata.contains("enriched-user"), "{metadata}");
}

#[rstest]
#[tokio::test]
async fn a_row_that_fails_to_build_keeps_its_counters(golden: (Json, Json)) {
    let (config, _) = golden;
    let harness = harness(Store::default());
    let mut request = success(&config, json!({"unbuildable": "boom"}));
    request["row"]["committed_tracked"] = json!(["spend:team:team-1"]);
    request["row"]["model_access_groups"] = json!(["group-a"]);

    let ran = run(
        &harness,
        input(Hook::AttemptEnded, request, priced(0.25), None),
        false,
    )
    .await;

    assert!(queued(&harness).is_empty());
    assert!(call(&ran.calls, "accounting_context")["built"].is_null());
    assert!(!names(&ran.calls).contains(&"after_write"));
    assert_eq!(counter(&harness, "spend:user:user-1"), Some(0.25));
    assert_eq!(
        counter(&harness, "spend:model_access_group:group-a"),
        Some(0.25)
    );
}

fn job_row(row: &Json) -> Json {
    let mut job = row.clone();
    job["call_type"] = json!("aretrieve_batch");
    job
}

#[rstest]
#[case::first_poll_owns_the_charge(Some(1), 1, true)]
#[case::a_later_poll_charges_nothing(Some(0), 0, false)]
#[tokio::test]
async fn a_settle_once_job_row_is_claimed_through_the_ledger_store(
    golden: (Json, Json),
    #[case] accepts: Option<usize>,
    #[case] charges: usize,
    #[case] claimed: bool,
) {
    let (config, row) = golden;
    let harness = harness(Store {
        accepts,
        ..Store::default()
    });
    let mut request = success(&config, job_row(&row));
    request["call_type"] = json!("aretrieve_batch");
    let mut hook = input(Hook::AttemptEnded, request, priced(0.5), None);
    hook.fields
        .insert(Field::SpendLogId, json!({"id": "batch_abc_batch_cost"}));

    let ran = run(&harness, hook, false).await;

    assert_eq!(harness.store.written.lock().unwrap().len(), 1);
    assert!(queued(&harness).is_empty());
    assert_eq!(
        names(&ran.calls)
            .iter()
            .filter(|name| **name == "after_charge")
            .count(),
        charges
    );
    let after_write = ran
        .calls
        .iter()
        .find(|(name, _)| name == "after_write")
        .map(|(_, request)| request["claimed"].clone());
    assert_eq!(after_write, claimed.then(|| json!(true)));
}

fn failure(config: &Json, rows: Vec<Json>) -> Json {
    json!({
        "kind": "failure",
        "settings": settings(config),
        "identity": identity(),
        "enrich": false,
        "track": true,
        "rows": rows.into_iter().map(|row| json!({"row": row})).collect::<Vec<_>>(),
    })
}

fn failure_input(
    request: Json,
    cost: Json,
    attempts: Json,
    reservation: Option<BudgetReservation>,
) -> HookInput {
    let mut hook = input(Hook::PostCallFailure, request, cost, reservation);
    if !attempts.is_null() {
        hook.fields.insert(Field::BilledRetryAttempts, attempts);
    }
    hook
}

#[rstest]
#[tokio::test]
async fn a_failure_releases_the_reservation_and_writes_every_billed_row(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());

    let ran = run(
        &harness,
        failure_input(
            failure(&config, vec![row.clone(), row]),
            json!({"failure_row_response_cost": 0.0}),
            json!([{"position": 0, "spend_log_id": {"id": "chatcmpl-attempt-0"}, "response_cost": 0.0}]),
            Some(reservation()),
        ),
        false,
    )
    .await;

    assert_eq!(counter(&harness, KEY_COUNTER), Some(3.0));
    let ids: Vec<Json> = queued(&harness)
        .iter()
        .map(|entry| row_json(entry)["request_id"].clone())
        .collect();
    assert_eq!(ids, [json!(REQUEST_ID), json!("chatcmpl-attempt-0")]);
    assert_eq!(
        names(&ran.calls),
        [
            "accounting_context",
            "after_write",
            "accounting_context",
            "after_write"
        ]
    );
    assert_eq!(ran.written_back.unwrap().finalized, Some(true));
}

#[rstest]
#[tokio::test]
async fn a_failure_with_a_cost_charges_the_counters_of_each_costed_row(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());

    run(
        &harness,
        failure_input(
            failure(&config, vec![row.clone(), row]),
            json!({"failure_row_response_cost": 0.25}),
            json!([{"position": 0, "spend_log_id": {"id": "chatcmpl-attempt-0"}, "response_cost": 0.5}]),
            None,
        ),
        false,
    )
    .await;

    let spends: Vec<Json> = queued(&harness)
        .iter()
        .map(|entry| row_json(entry)["spend"].clone())
        .collect();
    assert_eq!(spends, [json!(0.25), json!(0.5)]);
    assert_eq!(counter(&harness, "spend:user:user-1"), Some(0.75));
}

#[rstest]
#[tokio::test]
async fn a_completed_failure_cost_settles_the_reservation_like_a_success(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());

    let ran = run(
        &harness,
        failure_input(
            failure(&config, vec![row]),
            json!({"cost_finalized": true, "failure_row_response_cost": 0.5}),
            json!(null),
            Some(reservation()),
        ),
        false,
    )
    .await;

    assert_eq!(counter(&harness, KEY_COUNTER), Some(3.5));
    assert_eq!(counter(&harness, "spend:user:user-1"), Some(0.5));
    assert_eq!(ran.written_back.unwrap().finalized, Some(true));
}

#[rstest]
#[tokio::test]
async fn an_untracked_failure_only_releases(golden: (Json, Json)) {
    let (config, row) = golden;
    let harness = harness(Store::default());
    let mut request = failure(&config, vec![row]);
    request["track"] = json!(false);

    run(
        &harness,
        failure_input(request, json!({}), json!(null), Some(reservation())),
        false,
    )
    .await;

    assert_eq!(counter(&harness, KEY_COUNTER), Some(3.0));
    assert!(queued(&harness).is_empty());
}

#[tokio::test]
async fn a_run_without_the_host_is_refused() {
    let harness = harness(Store::default());
    let hook = input(Hook::AttemptEnded, json!({}), json!({}), None);

    assert!(harness.callback.handle(&hook, None).await.is_err());
}
