import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import pytest
import pytest_asyncio

import litellm
import litellm.proxy.proxy_server as ps
from litellm.integrations.custom_logger_v2.rust import RustLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.db.native_database import ServiceLoggingEvents
from litellm.proxy.db.rust_spend_log_writer import install_rust_spend_log_writer, uninstall_rust_spend_log_writer
from litellm.proxy.hooks.rust_spend_logger import bill_failure_in_rust, rust_spend_loggers, spend_logger
from litellm.proxy.utils import PrismaClient
from litellm.rust_bridge._native import DatabaseEndpoint, NativeDatabase, NativeSpendCounters, RustSpendLogWriter
from tests.unit.proxy.db.test_rust_spend_log_writer import FakePrismaClient

_UNREACHABLE: Final = "postgresql://litellm:secret@db.invalid:5432/proxy?pool_timeout=1"
_KEYED: Final[Mapping[str, object]] = {
    "user_api_key": "hashed-key",
    "user_api_key_alias": "alias",
    "user_api_key_user_id": "user-1",
    "user_api_key_team_id": "team-1",
    "user_api_key_org_id": "org-1",
}


@dataclass
class AfterWrite:
    calls: list[Mapping[str, object]]

    async def record(self, **call: object) -> None:  # kwargs-ok: records the post-write step's keyword arguments
        self.calls.append(call)


@dataclass(frozen=True, slots=True)
class Proxy:
    client: FakePrismaClient
    writer: RustSpendLogWriter
    logger: RustLogger
    after_write: AfterWrite

    async def rows(self, count: int = 1) -> Sequence[Mapping[str, object]]:
        for _ in range(300):
            if len(self.writer) >= count:
                break
            await asyncio.sleep(0.01)
        return [json.loads(row) for row in self.writer.dequeue(100)]


@pytest_asyncio.fixture(loop_scope="function")
async def proxy(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Proxy]:
    monkeypatch.setattr(PrismaClient, "spend_log_queue_bytes", 0)
    database: Final = await NativeDatabase.connect(
        ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=lambda: None),
        await DatabaseEndpoint.resolve(_UNREACHABLE, 1),
        None,
    )
    client: Final = FakePrismaClient()
    writer: Final = await install_rust_spend_log_writer(client, database, {"use_spend_logs_ledger": True}, None)  # pyright: ignore[reportArgumentType]  # the fake stands in for the Prisma client
    assert writer is not None
    handle: Final = ps.spend_counter_store.handle
    assert isinstance(handle, NativeSpendCounters)
    after_write: Final = AfterWrite([])
    monkeypatch.setattr(ps, "prisma_client", client)
    monkeypatch.setattr(ps, "general_settings", {"use_spend_logs_ledger": True})
    monkeypatch.setattr(ps.proxy_logging_obj.slack_alerting_instance, "alerting", [])
    monkeypatch.setattr(ps.proxy_logging_obj.db_spend_update_writer, "record_after_ledger_row", after_write.record)
    logger: Final = spend_logger(writer, handle)
    monkeypatch.setattr(litellm, "callbacks", [logger])
    try:
        yield Proxy(client, writer, logger, after_write)
    finally:
        await uninstall_rust_spend_log_writer(client)  # pyright: ignore[reportArgumentType]  # the fake stands in for the Prisma client
        ps.spend_counter_store.flush_local()


async def _complete(metadata: Mapping[str, object]) -> None:
    _ = await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello there",
        metadata=dict(metadata),
    )


def _auth(budget_reservation: dict[str, object] | None = None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="hashed-key",
        user_id="user-1",
        team_id="team-1",
        request_route="/v1/responses",
        budget_reservation=budget_reservation,
    )


def _metadata(row: Mapping[str, object]) -> Mapping[str, object]:
    metadata: Final = row["metadata"]
    assert isinstance(metadata, str)
    decoded: Final = json.loads(metadata)
    assert isinstance(decoded, dict)
    return decoded


@pytest.mark.asyncio
async def test_a_priced_completion_queues_one_ledger_row_and_runs_the_post_write_steps(proxy: Proxy) -> None:
    await _complete(_KEYED)

    rows: Final = await proxy.rows()

    assert len(rows) == 1
    assert rows[0]["status"] == "success"
    assert rows[0]["team_id"] == "team-1"
    assert isinstance(rows[0]["spend"], float) and rows[0]["spend"] > 0
    assert rows[0]["accounting_delivery_id"]
    for _ in range(100):
        if proxy.after_write.calls:
            break
        await asyncio.sleep(0.01)
    assert len(proxy.after_write.calls) == 1
    call: Final = proxy.after_write.calls[0]
    assert call["team_id"] == "team-1"
    assert call["spend_logs_kept"] is True
    assert call["claimed"] is False
    assert call["payload"]["request_id"] == rows[0]["request_id"]  # pyright: ignore[reportIndexIssue]  # the recorded payload dict


@pytest.mark.asyncio
async def test_a_success_writes_the_settled_reservation_back(proxy: Proxy) -> None:
    reservation: Final[dict[str, object]] = {"reserved_cost": 0.5, "entries": []}

    await _complete({**_KEYED, "user_api_key_budget_reservation": reservation})
    _ = await proxy.rows()
    for _ in range(300):
        if "finalized" in reservation:
            break
        await asyncio.sleep(0.01)

    assert reservation.get("finalized") is True


@pytest.mark.asyncio
async def test_a_failure_writes_the_released_reservation_back(proxy: Proxy) -> None:
    auth: Final = _auth(budget_reservation={"reserved_cost": 0.5, "entries": []})

    await bill_failure_in_rust(auth, ValueError("refused"), {"model": "gpt-4o", "metadata": {}})

    assert auth.budget_reservation is not None
    assert auth.budget_reservation.get("finalized") is True


@pytest.mark.asyncio
async def test_a_rejected_websocket_frame_queues_a_failure_row(proxy: Proxy) -> None:
    await bill_failure_in_rust(
        _auth(), ValueError("guardrail refused the frame"), {"model": "gpt-4o", "call_type": "aresponses"}
    )

    rows: Final = await proxy.rows()

    assert len(rows) == 1
    assert rows[0]["status"] == "failure"
    assert rows[0]["team_id"] == "team-1"


@pytest.mark.asyncio
async def test_a_failure_row_carries_the_callers_traceback(proxy: Proxy) -> None:
    _ = await ps.proxy_logging_obj.post_call_failure_hook(
        request_data={"model": "gpt-4o", "metadata": {}},
        original_exception=ValueError("upstream failed"),
        user_api_key_dict=_auth(),
        traceback_str="the caller's traceback",
    )

    rows: Final = await proxy.rows()

    assert len(rows) == 1
    error_information: Final = _metadata(rows[0])["error_information"]
    assert isinstance(error_information, dict)
    assert str(error_information.get("traceback", "")).startswith("the caller's traceback")


@pytest.mark.asyncio
async def test_a_responses_turn_replayed_from_the_database_wakes_the_monitor(proxy: Proxy) -> None:
    flush_requested: Final = asyncio.Event()
    proxy.client.spend_log_flush_requested = flush_requested

    _ = await litellm.aresponses(model="gpt-4o", input="hi", mock_response="hello there", metadata=dict(_KEYED))

    _ = await proxy.rows()
    await asyncio.wait_for(flush_requested.wait(), timeout=2)


@pytest.mark.asyncio
async def test_the_proxy_registers_the_rust_callback_only_over_a_rust_writer(proxy: Proxy) -> None:
    assert rust_spend_loggers(proxy.client) is not None
    assert rust_spend_loggers(FakePrismaClient()) is None
