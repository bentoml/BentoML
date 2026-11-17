from __future__ import annotations

import collections
import typing as t

import pytest

import bentoml
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.monitoring.base import MON_COLUMN_VAR
from bentoml._internal.monitoring.base import MON_DATAS_VAR
from bentoml._internal.monitoring.base import MonitorBase


class DummyMonitor(MonitorBase[int]):
    def __init__(self, name: str, **_: t.Any) -> None:
        super().__init__(name)
        self.exported_schema: dict[str, dict[str, str]] | None = None
        self.exported_data: dict[str, list[int]] | None = None

    def export_schema(self, columns_schema: dict[str, dict[str, str]]) -> None:
        self.exported_schema = columns_schema

    def export_data(self, datas: dict[str, collections.deque[int]]) -> None:
        self.exported_data = {name: list(values) for name, values in datas.items()}


@pytest.fixture(autouse=True)
def reset_monitor_state() -> t.Generator[None, None, None]:
    MON_COLUMN_VAR.set(None)
    MON_DATAS_VAR.set(None)
    yield
    MON_COLUMN_VAR.set(None)
    MON_DATAS_VAR.set(None)


@pytest.fixture(autouse=True)
def enable_monitoring() -> t.Generator[None, None, None]:
    enabled = BentoMLContainer.config.monitoring.enabled
    previous = enabled.get()
    enabled.set(True)
    yield
    enabled.set(previous)


def test_monitor_context_clears_state_when_nothing_is_logged() -> None:
    with bentoml.monitor("empty-context"):
        pass

    assert MON_COLUMN_VAR.get() is None
    assert MON_DATAS_VAR.get() is None

    # A second monitor block in the same context must still work.
    with bentoml.monitor("empty-context"):
        pass


def test_monitor_context_clears_state_when_body_raises() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with bentoml.monitor("raise-in-body"):
            raise RuntimeError("boom")

    assert MON_COLUMN_VAR.get() is None
    assert MON_DATAS_VAR.get() is None

    # A second monitor block in the same context must still work.
    with bentoml.monitor("raise-in-body"):
        pass


def test_stop_record_exports_data_and_resets_state() -> None:
    monitor = DummyMonitor("normal")
    monitor.start_record()
    monitor.log(1, "feature", "feature", "numerical")
    monitor.log(2, "feature", "feature", "numerical")
    monitor.stop_record()

    assert monitor.exported_schema == {
        "feature": {"name": "feature", "role": "feature", "type": "numerical"}
    }
    assert monitor.exported_data == {"feature": [1, 2]}
    assert MON_COLUMN_VAR.get() is None
    assert MON_DATAS_VAR.get() is None


def test_stop_record_rejects_mismatched_column_lengths() -> None:
    monitor = DummyMonitor("mismatched")
    monitor.start_record()
    monitor.log(1, "feature_a", "feature", "numerical")
    monitor.log(2, "feature_a", "feature", "numerical")
    monitor.log(3, "feature_b", "feature", "numerical")

    with pytest.raises(ValueError, match="same length"):
        monitor.stop_record()

    assert monitor.exported_data is None
    assert MON_COLUMN_VAR.get() is None
    assert MON_DATAS_VAR.get() is None
