//! Sums of `LiteLLM_SpendLogs.spend` for one key, team or user since a window start
//!
//! The spend log ledger's statements on `LiteLLM_SpendLogs` and `LiteLLM_SpendLogDelivery` sit here too,
//! mirroring the Python ledger writer (`SpendLogsRepository._insert_ledger`).

use chrono::NaiveDateTime;
use serde_json::Value;
use sqlx::PgExecutor;

use litellm_db::Error;

pub(super) async fn key_window_spend(
    executor: impl PgExecutor<'_>,
    api_key: &str,
    since: NaiveDateTime,
) -> Result<f64, Error> {
    Ok(
        sqlx::query_file_scalar!("src/spend_log/queries/key_window_spend.sql", api_key, since)
            .fetch_one(executor)
            .await?,
    )
}

pub(super) async fn team_window_spend(
    executor: impl PgExecutor<'_>,
    team_id: &str,
    since: NaiveDateTime,
) -> Result<f64, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/spend_log/queries/team_window_spend.sql",
        team_id,
        since
    )
    .fetch_one(executor)
    .await?)
}

pub(super) async fn user_window_spend(
    executor: impl PgExecutor<'_>,
    user_id: &str,
    since: NaiveDateTime,
) -> Result<f64, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/spend_log/queries/user_window_spend.sql",
        user_id,
        since
    )
    .fetch_one(executor)
    .await?)
}

/// Inserts the request logs in `records` (a JSON array of `LiteLLM_SpendLogs` rows), skipping
/// any `request_id` already stored, and returns the ids it inserted.
pub(super) async fn insert_spend_log_rows(
    executor: impl PgExecutor<'_>,
    records: Value,
) -> Result<Vec<String>, Error> {
    Ok(
        sqlx::query_file_scalar!("src/spend_log/queries/insert_spend_log_rows.sql", records)
            .fetch_all(executor)
            .await?,
    )
}

/// Inserts the deliveries in `payloads` (a JSON array of delivery rows), skipping any
/// `delivery_id` already stored, and returns the ids it accepted.
pub(super) async fn insert_spend_log_deliveries(
    executor: impl PgExecutor<'_>,
    payloads: Value,
) -> Result<Vec<String>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/spend_log/queries/insert_spend_log_deliveries.sql",
        payloads
    )
    .fetch_all(executor)
    .await?)
}

/// Which of the settle-once `request_ids` a pre-ledger cost row already charged.
pub(super) async fn spend_log_jobs_charged_by_legacy(
    executor: impl PgExecutor<'_>,
    request_ids: &[String],
) -> Result<Vec<String>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/spend_log/queries/select_spend_log_jobs_charged_by_legacy.sql",
        request_ids
    )
    .fetch_all(executor)
    .await?)
}

/// Removes the deliveries in `delivery_ids`.
pub(super) async fn delete_spend_log_deliveries(
    executor: impl PgExecutor<'_>,
    delivery_ids: &[String],
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/spend_log/queries/delete_spend_log_deliveries.sql",
        delivery_ids
    )
    .execute(executor)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use chrono::NaiveDateTime;
    use rstest::rstest;
    use sqlx::{PgConnection, PgPool, Postgres, Transaction};

    use super::{key_window_spend, team_window_spend, user_window_spend};
    use crate::test_support::database_pool;

    #[derive(Clone, Copy, Debug)]
    enum Column {
        Key,
        Team,
        User,
    }

    async fn window_spend(
        column: Column,
        executor: &mut PgConnection,
        owner_id: &str,
        since: NaiveDateTime,
    ) -> Result<f64, litellm_db::Error> {
        match column {
            Column::Key => key_window_spend(executor, owner_id, since).await,
            Column::Team => team_window_spend(executor, owner_id, since).await,
            Column::User => user_window_spend(executor, owner_id, since).await,
        }
    }

    /// One log owned by `owner` in the column under test, with `other` in the other two owner columns. The column
    /// goes into the request id so the cases, which run at once, never wait on each other's primary key.
    async fn insert_spend_log(
        tx: &mut Transaction<'_, Postgres>,
        column: Column,
        request_id: &str,
        owner: &str,
        other: &str,
        spend: f64,
        start_time: &str,
    ) -> Result<(), sqlx::Error> {
        let (api_key, team_id, user) = match column {
            Column::Key => (owner, other, other),
            Column::Team => (other, owner, other),
            Column::User => (other, other, owner),
        };
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_SpendLogs"
                (request_id, call_type, api_key, team_id, "user", spend, "startTime", "endTime")
            VALUES ($1, 'acompletion', $2, $3, $4, $5, $6::timestamp, $6::timestamp)
            "#,
        )
        .bind(format!("{request_id}-{column:?}"))
        .bind(api_key)
        .bind(team_id)
        .bind(user)
        .bind(spend)
        .bind(start_time)
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    fn window_start() -> Result<NaiveDateTime, chrono::ParseError> {
        NaiveDateTime::parse_from_str("2026-08-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    }

    #[rstest]
    #[case::key(Column::Key)]
    #[case::team(Column::Team)]
    #[case::user(Column::User)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn sums_only_the_owners_column_from_the_window_start_on(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
        #[case] column: Column,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        for (request_id, owner, other, spend, start_time) in [
            (
                "window-before",
                "window-owner",
                "window-x",
                100.0,
                "2026-07-31 23:59:59.999",
            ),
            (
                "window-at-start",
                "window-owner",
                "window-x",
                1.25,
                "2026-08-01 00:00:00",
            ),
            (
                "window-inside",
                "window-owner",
                "window-x",
                2.5,
                "2026-08-03 12:00:00",
            ),
            (
                "window-other-column",
                "window-x",
                "window-owner",
                50.0,
                "2026-08-02 00:00:00",
            ),
        ] {
            insert_spend_log(&mut tx, column, request_id, owner, other, spend, start_time).await?;
        }

        assert_eq!(
            window_spend(column, &mut tx, "window-owner", window_start()?).await?,
            3.75
        );
        Ok(())
    }

    #[rstest]
    #[case::key(Column::Key)]
    #[case::team(Column::Team)]
    #[case::user(Column::User)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn no_rows_in_the_window_is_zero(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
        #[case] column: Column,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_spend_log(
            &mut tx,
            column,
            "window-old",
            "window-quiet-owner",
            "window-x",
            9.0,
            "2026-07-01 00:00:00",
        )
        .await?;
        let since = window_start()?;

        assert_eq!(
            window_spend(column, &mut tx, "window-quiet-owner", since).await?,
            0.0
        );
        assert_eq!(
            window_spend(column, &mut tx, "window-missing-owner", since).await?,
            0.0
        );
        Ok(())
    }
}
