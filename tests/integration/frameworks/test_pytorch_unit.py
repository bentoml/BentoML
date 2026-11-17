from __future__ import annotations

import numpy as np
import pytest
import torch

from bentoml._internal.frameworks.common.pytorch import make_pytorch_runnable_method
from bentoml._internal.frameworks.pytorch import PyTorchTensorContainer
from bentoml._internal.runner.container import AutoContainer


@pytest.mark.parametrize("batch_axis", [0, 1])
def test_pytorch_container(batch_axis: int):
    one_batch = torch.arange(6).reshape(2, 3)
    batch_list = [one_batch, one_batch + 1]
    merged_batch = torch.cat(batch_list, dim=batch_axis)

    batches, indices = PyTorchTensorContainer.batches_to_batch(
        batch_list,
        batch_dim=batch_axis,
    )
    assert batches.shape == merged_batch.shape
    assert (batches == merged_batch).all()
    assert (
        PyTorchTensorContainer.batch_to_batches(
            merged_batch,
            indices=indices,
            batch_dim=batch_axis,
        )[0]
        == one_batch
    ).all()

    assert (
        PyTorchTensorContainer.from_payload(
            PyTorchTensorContainer.to_payload(one_batch)
        )
        == one_batch
    ).all()

    assert (
        AutoContainer.from_payload(AutoContainer.to_payload(one_batch, batch_dim=0))
        == one_batch
    ).all()


class _IdentityModel(torch.nn.Module):
    """Minimal nn.Module that returns its first input — used to inspect the
    tensor the runnable hands to the model after numpy/pandas conversion."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeRunnable:
    """A stand-in for PytorchModelRunnable that lets us drive
    make_pytorch_runnable_method without a saved bento model."""

    def __init__(self) -> None:
        self.device_id = "cpu"
        self.model = _IdentityModel()


@pytest.mark.parametrize(
    "np_dtype, expected_torch_dtype",
    [
        (np.float16, torch.float16),
        (np.float32, torch.float32),
        (np.float64, torch.float64),
        (np.int8, torch.int8),
        (np.int32, torch.int32),
        (np.int64, torch.int64),
        (np.uint8, torch.uint8),
        (np.bool_, torch.bool),
    ],
)
def test_pytorch_runnable_method_preserves_numpy_dtype(
    np_dtype: np.dtype, expected_torch_dtype: torch.dtype
) -> None:
    """Regression for #4266: the numpy -> torch conversion inside the pytorch
    runnable used ``torch.Tensor(arr)``, which silently upcast every input to
    ``float32`` and broke models whose weights were ``float16``, ``int64``,
    etc."""
    runnable = _FakeRunnable()
    method = make_pytorch_runnable_method("forward")
    arr = (
        np.array([True, False, True], dtype=np_dtype)
        if np_dtype is np.bool_
        else np.array([1, 2, 3], dtype=np_dtype)
    )

    result = method(runnable, arr)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == expected_torch_dtype


def test_pytorch_runnable_method_preserves_pandas_dtype() -> None:
    """Same dtype-preservation contract for pandas DataFrame inputs."""
    pd = pytest.importorskip("pandas")
    runnable = _FakeRunnable()
    method = make_pytorch_runnable_method("forward")
    df = pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)})

    result = method(runnable, df)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.int64
