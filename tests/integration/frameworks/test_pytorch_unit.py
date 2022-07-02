from __future__ import annotations

import torch
import pytest

from bentoml._internal.runner.container import AutoContainer
from bentoml._internal.frameworks.pytorch import PyTorchTensorContainer


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
