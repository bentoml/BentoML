from __future__ import annotations

import asyncio
import typing as t
from functools import partial

import numpy as np
import pandas as pd
import pytest

import bentoml._internal.runner.container as c
from bentoml._internal.marshal.dispatcher import CorkDispatcher


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


@pytest.mark.parametrize("rows, columns", [(0, 2), (1, 2), (3, 0), (5, 2)])
def test_pandas_batch_size_counts_rows(rows: int, columns: int):
    batch = pd.DataFrame(np.zeros((rows, columns)))

    assert c.AutoContainer.get_batch_size(batch, batch_dim=0) == rows


@pytest.mark.asyncio
async def test_pandas_dispatcher_splits_oversized_requests():
    max_batch_size = 2
    dispatcher = CorkDispatcher(
        max_latency_in_ms=1000,
        max_batch_size=max_batch_size,
        get_batch_size=partial(c.AutoContainer.get_batch_size, batch_dim=0),
        fallback=lambda: pytest.fail("DataFrame request was dropped"),
    )
    batch_sizes = []

    async def infer(batches):
        batch, indices = c.AutoContainer.batches_to_batch(batches)
        batch_sizes.append(len(batch))
        return c.AutoContainer.batch_to_batches(batch * 2, indices)

    inputs = [pd.DataFrame({"value": [1, 2, 3, 4, 5]}), pd.DataFrame({"value": [9]})]
    dispatch = dispatcher(infer)
    try:
        outputs = await asyncio.wait_for(
            asyncio.gather(*(dispatch(batch) for batch in inputs)), timeout=5
        )
        for actual, expected in zip(outputs, inputs):
            pd.testing.assert_frame_equal(actual, expected * 2)
        assert batch_sizes
        assert max(batch_sizes) <= max_batch_size
        assert sum(batch_sizes) == sum(len(batch) for batch in inputs)
    finally:
        dispatcher.shutdown()
        if dispatcher._controller is not None:
            await asyncio.gather(dispatcher._controller, return_exceptions=True)
