"""Unit tests for task progress tracking (Issue #5434)."""

from __future__ import annotations

import datetime

import pytest

from _bentoml_impl.tasks.result import ResultRow
from _bentoml_impl.tasks.result import ResultStatus
from _bentoml_impl.tasks.result import Sqlite3Store

# ---------------------------------------------------------------------------
# ResultRow field tests
# ---------------------------------------------------------------------------


def make_row(**kwargs) -> ResultRow:
    defaults = dict(
        task_id="abc123",
        name="my_task",
        status=ResultStatus.IN_PROGRESS,
        created_at=datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        executed_at=None,
    )
    defaults.update(kwargs)
    return ResultRow(**defaults)


def test_progress_defaults_to_none():
    row = make_row()
    assert row.progress is None


def test_progress_field_included_in_to_json():
    row = make_row()
    data = row.to_json()
    assert "progress" in data
    assert data["progress"] is None


def test_progress_value_in_to_json():
    row = make_row(progress=0.75)
    data = row.to_json()
    assert data["progress"] == 0.75


def test_progress_can_be_set_on_construction():
    row = make_row(progress=0.5)
    assert row.progress == 0.5


# ---------------------------------------------------------------------------
# Sqlite3Store.update_progress tests (uses a real in-memory-like temp DB)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path):
    db_file = str(tmp_path / "test_results.db")
    Sqlite3Store.init_db(db_file)
    return db_file


@pytest.mark.asyncio
async def test_update_progress_sets_value(tmp_db):

    async with Sqlite3Store(tmp_db) as store:
        # Manually insert a task entry so we can test update_progress

        task_id = "test-task-001"
        await store._conn.execute(
            "INSERT INTO result (task_id, name, input, status) VALUES (?, ?, ?, ?)",
            (task_id, "demo", b"{}", ResultStatus.IN_PROGRESS.value),
        )
        await store._conn.commit()

        # Update progress to 0.5
        await store.update_progress(task_id, 0.5)

        row = await store.get_status(task_id)
        assert row.progress == 0.5


@pytest.mark.asyncio
async def test_update_progress_boundary_values(tmp_db):
    async with Sqlite3Store(tmp_db) as store:
        task_id = "test-task-002"
        await store._conn.execute(
            "INSERT INTO result (task_id, name, input, status) VALUES (?, ?, ?, ?)",
            (task_id, "demo", b"{}", ResultStatus.IN_PROGRESS.value),
        )
        await store._conn.commit()

        # Boundary values should succeed
        await store.update_progress(task_id, 0.0)
        row = await store.get_status(task_id)
        assert row.progress == 0.0

        await store.update_progress(task_id, 1.0)
        row = await store.get_status(task_id)
        assert row.progress == 1.0


@pytest.mark.asyncio
async def test_update_progress_rejects_out_of_range(tmp_db):
    async with Sqlite3Store(tmp_db) as store:
        task_id = "test-task-003"
        await store._conn.execute(
            "INSERT INTO result (task_id, name, input, status) VALUES (?, ?, ?, ?)",
            (task_id, "demo", b"{}", ResultStatus.IN_PROGRESS.value),
        )
        await store._conn.commit()

        with pytest.raises(ValueError, match="0.0 and 1.0"):
            await store.update_progress(task_id, -0.1)

        with pytest.raises(ValueError, match="0.0 and 1.0"):
            await store.update_progress(task_id, 1.1)
