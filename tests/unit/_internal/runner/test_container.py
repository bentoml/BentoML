from __future__ import annotations

import subprocess
import sys
import typing as t

import numpy as np
import pandas as pd
import pytest

import bentoml._internal.runner.container as c


@pytest.mark.parametrize("batch_dim_exc", [AssertionError])
@pytest.mark.parametrize("wrong_batch_dim", [1, 19])
def test_default_container(batch_dim_exc: t.Type[Exception], wrong_batch_dim: int):
    l1 = [1, 2, 3]
    l2 = [3, 4, 5, 6]
    batch, indices = c.DefaultContainer.batches_to_batch([l1, l2])
    assert batch == l1 + l2
    assert indices == [0, 3, 7]
    restored_l1, restored_l2 = c.DefaultContainer.batch_to_batches(batch, indices)
    assert restored_l1 == l1
    assert restored_l2 == l2

    # DefaultContainer should only allow batch_dim = 0
    with pytest.raises(batch_dim_exc):
        c.DefaultContainer.batches_to_batch([l1, l2], batch_dim=wrong_batch_dim)

    with pytest.raises(batch_dim_exc):
        c.DefaultContainer.batch_to_batches(batch, indices, batch_dim=wrong_batch_dim)

    def _generator():
        yield "apple"
        yield "banana"
        yield "cherry"

    assert c.DefaultContainer.from_payload(
        c.DefaultContainer.to_payload(_generator(), batch_dim=0)
    ) == list(_generator())

    assert c.DefaultContainer.from_batch_payloads(
        c.DefaultContainer.batch_to_payloads(batch, indices)
    ) == (batch, indices)


@pytest.mark.parametrize("batch_dim", [0, 1])
def test_ndarray_container(batch_dim: int):
    arr1 = np.ones((3, 3))
    if batch_dim == 0:
        arr2 = np.arange(6).reshape(2, 3)
    else:
        arr2 = np.arange(6).reshape(3, 2)

    batches = [arr1, arr2]
    batch, indices = c.NdarrayContainer.batches_to_batch(batches, batch_dim=batch_dim)
    assert (batch == np.concatenate(batches, axis=batch_dim)).all()
    restored_arr1, restored_arr2 = c.NdarrayContainer.batch_to_batches(
        batch, indices, batch_dim=batch_dim
    )
    assert (arr1 == restored_arr1).all()
    assert (arr2 == restored_arr2).all()

    assert (
        c.NdarrayContainer.from_payload(c.NdarrayContainer.to_payload(arr1, batch_dim))
        == arr1
    ).all()

    restored_batch, restored_indices = c.NdarrayContainer.from_batch_payloads(
        c.NdarrayContainer.batch_to_payloads(batch, indices, batch_dim=batch_dim),
        batch_dim=batch_dim,
    )
    assert restored_indices == indices
    assert (restored_batch == batch).all()


@pytest.mark.parametrize("batch_dim_exc", [AssertionError])
@pytest.mark.parametrize("wrong_batch_dim", [1, 19])
def test_pandas_container(batch_dim_exc: t.Type[Exception], wrong_batch_dim: int):
    cols = ["a", "b", "c"]
    arr1 = np.ones((3, 3))
    df1 = pd.DataFrame(arr1, columns=cols)
    arr2 = np.arange(6, dtype=np.float64).reshape(2, 3)
    df2 = pd.DataFrame(arr2, columns=cols)
    batches = [df1, df2]
    batch, indices = c.PandasDataFrameContainer.batches_to_batch(batches)
    assert batch.equals(pd.concat(batches, ignore_index=True))

    restored_df1, restored_df2 = c.PandasDataFrameContainer.batch_to_batches(
        batch, indices
    )
    assert df1.equals(restored_df1)
    assert df2.equals(restored_df2)

    assert c.PandasDataFrameContainer.from_payload(
        c.PandasDataFrameContainer.to_payload(df1, batch_dim=0)
    ).equals(df1)

    restored_batch, restored_indices = c.PandasDataFrameContainer.from_batch_payloads(
        c.PandasDataFrameContainer.batch_to_payloads(batch, indices)
    )
    assert restored_indices == indices
    assert restored_batch.equals(batch)

    # PandasDataFrameContainer should only allow batch_dim = 0

    with pytest.raises(batch_dim_exc):
        c.PandasDataFrameContainer.batches_to_batch(batches, batch_dim=wrong_batch_dim)

    with pytest.raises(batch_dim_exc):
        c.PandasDataFrameContainer.batch_to_batches(
            batch, indices, batch_dim=wrong_batch_dim
        )


TORCH_BATCH_SCRIPT = """
import sys
import torch
from bentoml._internal.runner.container import AutoContainer

batch_dim = int(sys.argv[1])
batches = [torch.ones(int(r), int(c)) for r, c in (s.split("x") for s in sys.argv[2:])]
batch, indices = AutoContainer.batches_to_batch(batches, batch_dim)
restored = AutoContainer.batch_to_batches(batch, indices, batch_dim)
print(type(batch).__name__, AutoContainer.get_batch_size(batches[0], batch_dim))
print([tuple(r.shape) for r in restored] == [tuple(b.shape) for b in batches])
"""


@pytest.mark.parametrize(
    "batch_dim, shapes, expected",
    [
        (0, ["2x2", "1x2"], ["Tensor", "2", "True"]),
        (1, ["2x3", "2x1"], ["Tensor", "3", "True"]),
    ],
)
def test_torch_tensor_batches_without_framework_import(
    batch_dim: int, shapes: list[str], expected: list[str]
):
    # New-style services batch torch.Tensor inputs without ever importing
    # bentoml.pytorch; run in a fresh interpreter so nothing has imported it.
    pytest.importorskip("torch")
    result = subprocess.run(
        [sys.executable, "-c", TORCH_BATCH_SCRIPT, str(batch_dim), *shapes],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split() == expected
