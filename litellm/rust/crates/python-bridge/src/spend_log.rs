//! Python binding for the Rust spend log callback (`litellm-callbacks-spend-logger`). The proxy
//! builds one [`RustSpendLog`] over the installed Rust spend log writer, the shared spend counters
//! and the process's token redactor, and registers its `.callback()` as a Rust `CustomLoggerV2`.

use std::sync::Arc;

use litellm_callbacks::ErasedCallback;
use litellm_callbacks_budgets::Budgets;
use litellm_callbacks_python::RustCallback;
use litellm_callbacks_spend_logger::{SpendLogCallback, SpendLogParts};
use litellm_spend_log_writer::{PgLedgerStore, Retention};
use pyo3::prelude::*;

use crate::{
    logger::NativeTokenRedactor,
    redis::store::spend_counters::NativeSpendCounters,
    spend_log_writer::{RetentionFields, RustSpendLogWriter},
};

/// The spend log callback over the writer the proxy's monitor flushes and the spend counters
/// admission reads.
#[pyclass(frozen, name = "RustSpendLog")]
pub(crate) struct RustSpendLog(Arc<SpendLogCallback<PgLedgerStore>>);

#[pymethods]
impl RustSpendLog {
    /// `retention` is `(include, exclude)` from `spend_logs_metadata_fields`, for the job cost
    /// rows the callback claims outside the queue.
    #[new]
    #[pyo3(signature = (writer, spend_counters, token_redactor, retention=None))]
    fn new(
        py: Python<'_>,
        writer: PyRef<'_, RustSpendLogWriter>,
        spend_counters: PyRef<'_, NativeSpendCounters>,
        token_redactor: PyRef<'_, NativeTokenRedactor>,
        retention: Option<RetentionFields>,
    ) -> Self {
        Self(Arc::new(SpendLogCallback::new(SpendLogParts {
            writer: writer.shared(),
            budgets: Arc::new(
                Budgets::new(spend_counters.store()).with_logger(crate::logger::capture(py)),
            ),
            tokens: token_redactor.shared(),
            retention: retention.map(|(include, exclude)| Retention { include, exclude }),
        })))
    }

    fn callback(&self) -> PyResult<RustCallback> {
        Ok(RustCallback::new(
            Arc::clone(&self.0) as Arc<dyn ErasedCallback>
        )?)
    }
}
