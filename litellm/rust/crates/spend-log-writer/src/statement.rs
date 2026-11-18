//! Turning a queued request into what the repository writes: the typed `LiteLLM_SpendLogs` record
//! with its metadata cut to the retained fields, and its `LiteLLM_SpendLogDelivery` row.

use chrono::{DateTime, NaiveDateTime};
use litellm_callbacks_spend_log::{SpendLogRow, SpendLogStatus};
use litellm_python_compat::{
    json,
    safe_json::{SafeJsonOptions, safe_json_structure},
};
use litellm_repositories::{SpendLogDelivery, SpendLogRecord, SpendLogWrite};
use serde_json::Value;

use crate::QueuedSpendLog;

/// `SPEND_LOGS_METADATA_ALWAYS_KEPT_FIELDS`: metadata fields no retention setting drops.
const ALWAYS_KEPT: [&str; 2] = ["cold_storage_object_key", "status"];

/// Which metadata fields a logged row keeps (`general_settings.spend_logs_metadata_fields`),
/// validated in Python to set exactly one of the two lists.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Retention {
    pub include: Option<Vec<String>>,
    pub exclude: Option<Vec<String>>,
}

impl Retention {
    /// `SpendLogsMetadataFields.keeps(name)`.
    pub fn keeps(&self, name: &str) -> bool {
        if ALWAYS_KEPT.contains(&name) {
            return true;
        }
        match (&self.include, &self.exclude) {
            (Some(include), _) => include.iter().any(|field| field == name),
            (None, exclude) => !exclude.iter().flatten().any(|field| field == name),
        }
    }
}

/// A queued row the `LiteLLM_SpendLogs` table cannot hold, refused before it reaches Postgres.
#[derive(Debug, thiserror::Error)]
pub enum RowError {
    #[error("Spend log column {0} is not valid JSON")]
    InvalidJson(&'static str),
    #[error("Spend log column {0} is not an ISO 8601 timestamp")]
    InvalidTimestamp(&'static str),
    #[error("Spend log column {0} is out of range")]
    OutOfRange(&'static str),
    #[error("Spend log column spend is null")]
    NullSpend,
}

fn timestamp(column: &'static str, text: &str) -> Result<NaiveDateTime, RowError> {
    DateTime::parse_from_rfc3339(text)
        .map(|timestamp| timestamp.naive_utc())
        .or_else(|_| NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f"))
        .map_err(|_| RowError::InvalidTimestamp(column))
}

fn int(column: &'static str, value: i64) -> Result<i32, RowError> {
    i32::try_from(value).map_err(|_| RowError::OutOfRange(column))
}

/// The JSON text of a `jsonb` column as the value Postgres stores: non-finite numbers become
/// null, NUL characters are dropped and nesting past `safe_dumps`'s depth is marked, as the
/// Python insert's `safe_dumps` does.
fn json_column(column: &'static str, text: &str) -> Result<Value, RowError> {
    json::loads(text)
        .and_then(|value| {
            json::to_json(&safe_json_structure(
                &value,
                SafeJsonOptions {
                    non_finite_as_null: true,
                    integer_limit: None,
                },
            ))
        })
        .map_err(|_| RowError::InvalidJson(column))
}

fn retained(metadata: Value, retention: Option<&Retention>) -> Value {
    match (metadata, retention) {
        (Value::Object(fields), Some(retention)) => Value::Object(
            fields
                .into_iter()
                .filter(|(name, _)| retention.keeps(name))
                .collect(),
        ),
        (metadata, _) => metadata,
    }
}

fn record(
    row: &SpendLogRow,
    retention: Option<&Retention>,
    now: NaiveDateTime,
) -> Result<SpendLogRecord, RowError> {
    Ok(SpendLogRecord {
        request_id: row.request_id.clone(),
        call_type: row.call_type.clone(),
        api_key: row.api_key.clone(),
        spend: row.spend.ok_or(RowError::NullSpend)?,
        total_tokens: int("total_tokens", row.total_tokens)?,
        prompt_tokens: int("prompt_tokens", row.prompt_tokens)?,
        completion_tokens: int("completion_tokens", row.completion_tokens)?,
        start_time: timestamp("startTime", &row.start_time)?,
        end_time: timestamp("endTime", &row.end_time)?,
        request_duration_ms: row
            .request_duration_ms
            .map(|value| int("request_duration_ms", value))
            .transpose()?,
        completion_start_time: row
            .completion_start_time
            .as_deref()
            .map(|text| timestamp("completionStartTime", text))
            .transpose()?,
        model: row.model.clone(),
        model_id: Some(row.model_id.clone()),
        model_group: Some(row.model_group.clone()),
        custom_llm_provider: Some(row.custom_llm_provider.clone()),
        api_base: Some(row.api_base.clone()),
        user: Some(row.user.clone()),
        metadata: Some(retained(json_column("metadata", &row.metadata)?, retention)),
        cache_hit: Some(row.cache_hit.clone()),
        cache_key: Some(row.cache_key.clone()),
        request_tags: Some(json_column("request_tags", &row.request_tags)?),
        team_id: Some(row.team_id.clone()),
        organization_id: Some(row.organization_id.clone()),
        end_user: Some(row.end_user.clone()),
        requester_ip_address: row.requester_ip_address.clone(),
        messages: Some(json_column("messages", &row.messages)?),
        response: Some(json_column("response", &row.response)?),
        session_id: row.session_id.clone(),
        status: Some(
            match row.status {
                SpendLogStatus::Success => "success",
                SpendLogStatus::Failure => "failure",
            }
            .to_owned(),
        ),
        mcp_namespaced_tool_name: row.mcp_namespaced_tool_name.clone(),
        billing_agent_id: row.billing_agent_id.clone(),
        agent_id: row.agent_id.clone(),
        proxy_server_request: Some(json_column(
            "proxy_server_request",
            &row.proxy_server_request,
        )?),
        litellm_call_id: row.litellm_call_id.clone(),
        created_at: now,
        updated_at: now,
    })
}

/// The delivery a queued request's charge writes, or `None` when it carries no charge. A
/// settle-once job's delivery takes its request id.
pub fn delivery(entry: &QueuedSpendLog) -> Option<SpendLogDelivery> {
    let charge = entry.charge()?;
    Some(SpendLogDelivery {
        delivery_id: charge
            .delivery_id
            .clone()
            .unwrap_or_else(|| entry.request_id().to_owned()),
        request_id: entry.request_id().to_owned(),
        accounting_payload: charge.payload.clone(),
        is_batch: charge.delivery_id.is_none(),
    })
}

/// What one queued request writes, stamped with `now` as its `created_at` and `updated_at`.
pub fn spend_log_write(
    entry: &QueuedSpendLog,
    retention: Option<&Retention>,
    now: NaiveDateTime,
) -> Result<SpendLogWrite, RowError> {
    Ok(SpendLogWrite {
        log: entry
            .row()
            .map(|row| record(row, retention, now))
            .transpose()?,
        delivery: delivery(entry),
    })
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDateTime;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::{Retention, RowError, spend_log_write};
    use crate::{QueuedSpendLog, test_support::python_row};

    fn now() -> NaiveDateTime {
        NaiveDateTime::parse_from_str("2026-11-02T12:00:05", "%Y-%m-%dT%H:%M:%S").expect("now")
    }

    fn entry(fields: &str) -> QueuedSpendLog {
        QueuedSpendLog::from_python(&python_row("r", fields)).expect("row is queued")
    }

    #[rstest]
    #[case::include(Retention { include: Some(vec!["a".into()]), exclude: None }, json!({"a": 1, "status": "success"}))]
    #[case::exclude(Retention { include: None, exclude: Some(vec!["a".into()]) }, json!({"b": 2, "status": "success"}))]
    fn retains_configured_metadata_fields(#[case] retention: Retention, #[case] expected: Value) {
        let write = spend_log_write(
            &entry(r#", "metadata": "{\"a\": 1, \"b\": 2, \"status\": \"success\"}""#),
            Some(&retention),
            now(),
        )
        .expect("writes");

        assert_eq!(write.log.and_then(|log| log.metadata), Some(expected));
    }

    #[rstest]
    #[case::not_an_object(r#", "metadata": "[1]""#, json!([1]))]
    #[case::nan_and_nul(r#", "metadata": "{\"spend\": NaN, \"a\\u0000\": \"x\\u0000y\"}""#, json!({"spend": null, "a": "xy"}))]
    fn stores_metadata_as_safe_dumps_writes_it(#[case] fields: &str, #[case] expected: Value) {
        let write =
            spend_log_write(&entry(fields), Some(&Retention::default()), now()).expect("writes");

        assert_eq!(write.log.and_then(|log| log.metadata), Some(expected));
    }

    #[rstest]
    #[case::rfc3339_offset("2026-11-02T14:00:00+02:00", "2026-11-02T12:00:00")]
    #[case::naive_iso("2026-11-02T12:00:00.5", "2026-11-02T12:00:00.500")]
    fn reads_iso_timestamps(#[case] text: &str, #[case] expected: &str) {
        let write = spend_log_write(&entry(&format!(r#", "startTime": "{text}""#)), None, now())
            .expect("writes");

        assert_eq!(
            write.log.map(|log| log.start_time),
            Some(NaiveDateTime::parse_from_str(expected, "%Y-%m-%dT%H:%M:%S%.f").expect("parses"))
        );
    }

    #[rstest]
    #[case::invalid_json(r#", "messages": "not json""#)]
    #[case::not_a_time(r#", "startTime": "soon""#)]
    #[case::too_many_tokens(r#", "total_tokens": 9999999999"#)]
    #[case::null_spend(r#", "spend": null"#)]
    fn refuses_a_row_postgres_cannot_hold(#[case] fields: &str) {
        assert!(matches!(
            spend_log_write(&entry(fields), None, now()),
            Err(RowError::InvalidJson(_)
                | RowError::InvalidTimestamp(_)
                | RowError::OutOfRange(_)
                | RowError::NullSpend)
        ));
    }

    #[rstest]
    #[case::ledger_row(r#", "accounting_delivery_id": "d", "accounting_payload": {"spend": 1.0}"#, Some(("d", false)), true)]
    #[case::settle_once_job(r#", "accounting_payload": {"spend": 1.0}"#, Some(("r", true)), true)]
    #[case::uncharged("", None, true)]
    fn writes_the_delivery_the_row_carries(
        #[case] fields: &str,
        #[case] expected: Option<(&str, bool)>,
        #[case] logged: bool,
    ) {
        let write = spend_log_write(&entry(fields), None, now()).expect("writes");

        assert_eq!(
            write
                .delivery
                .as_ref()
                .map(|delivery| (delivery.delivery_id.as_str(), delivery.is_batch)),
            expected
        );
        assert_eq!(write.log.is_some(), logged);
    }
}
