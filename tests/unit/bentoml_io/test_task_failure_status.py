from __future__ import annotations

import pytest
from starlette.requests import Request

from _bentoml_impl.server.app import ServiceAppFactory
from _bentoml_impl.tasks.result import ResultStatus
from _bentoml_impl.tasks.result import Sqlite3Store
from _bentoml_sdk import service
from _bentoml_sdk import task


def _make_request() -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/process",
            "raw_path": b"/process",
            "query_string": b"",
            "headers": [(b"host", b"testserver")],
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        },
        receive,
    )


@pytest.mark.asyncio
async def test_run_task_persists_failed_status_when_endpoint_raises(
    tmp_path, monkeypatch
):
    @service
    class S:
        @task
        def process(self, x: str) -> str:
            return x

    factory = ServiceAppFactory(
        service=S,
        is_main=True,
        enable_metrics=False,
        services={S.name: {"traffic": {}, "workers": None}},
        enable_access_control=False,
        access_control_options={},
    )
    db_file = tmp_path / "results.db"
    Sqlite3Store.init_db(str(db_file))
    store = Sqlite3Store(str(db_file))
    factory._result_store = store

    async def boom(name, request):
        raise RuntimeError("endpoint exploded")

    monkeypatch.setattr(factory, "api_endpoint_wrapper", boom)

    request = _make_request()
    async with store:
        with S.context.in_request(request):
            task_id = await store.new_entry("process", request)
            await factory._run_task(task_id, "process", request)
        row = await store.get(task_id)

    assert row.status == ResultStatus.FAILED
    assert row.completed_at is not None
    assert row.result.status_code == 500
