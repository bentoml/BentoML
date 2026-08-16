from __future__ import annotations

import pytest
from starlette.requests import Request

from _bentoml_impl.server.app import ServiceAppFactory
from _bentoml_impl.tasks.result import ResultStatus
from _bentoml_impl.tasks.result import Sqlite3Store
from _bentoml_sdk import service
from _bentoml_sdk import task


def _make_request(query: dict[str, str]) -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    query_string = "&".join(f"{k}={v}" for k, v in query.items()).encode()
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "PUT",
            "scheme": "http",
            "path": "/process/cancel",
            "raw_path": b"/process/cancel",
            "query_string": query_string,
            "headers": [(b"host", b"testserver")],
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        },
        receive,
    )


@pytest.mark.asyncio
async def test_cancel_task_does_not_mutate_status(tmp_path, monkeypatch):
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

    request = _make_request({})
    async with store:
        with S.context.in_request(request):
            task_id = await store.new_entry("process", request)

        cancel_request = _make_request({"task_id": task_id})
        resp = await factory.cancel_task(cancel_request)

        assert resp.status_code == 400
        row = await store.get_status(task_id)
        assert row.status == ResultStatus.IN_PROGRESS
