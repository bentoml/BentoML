from __future__ import annotations

import typing as t

import pandas as pd
import pytest

if t.TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_log_collection(host: str, monitoring_dir: Path):
    data_path = monitoring_dir.joinpath("iris_classifier_prediction", "data")
    assert monitoring_dir.exists()
    assert data_path.exists()
    assert (
        pd.concat(
            [pd.read_json(f.__fspath__(), lines=True) for f in data_path.glob("*")]
        )
        is not None
    )
