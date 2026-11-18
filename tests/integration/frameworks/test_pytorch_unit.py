from __future__ import annotations

import pytest
import torch

import bentoml
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.frameworks.pytorch import PyTorchTensorContainer
from bentoml._internal.models import ModelStore
from bentoml._internal.runner.container import AutoContainer


class _Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


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


def test_load_model_defaults_to_weights_only_false(tmp_path):
    # Regression test for #5365: PyTorch >= 2.6 defaults `torch.load` to
    # `weights_only=True`, which cannot unpickle the whole-model artifact that
    # `save_model` writes via cloudpickle. `load_model` must default to
    # `weights_only=False` so a trusted, BentoML-produced model loads correctly,
    # while still honoring an explicit override passed by the caller.
    BentoMLContainer.model_store.set(ModelStore(str(tmp_path)))
    try:
        saved = bentoml.pytorch.save_model("weights_only_model", _Net())

        loaded = bentoml.pytorch.load_model(saved)
        assert isinstance(loaded, torch.nn.Module)

        with pytest.raises(Exception):
            bentoml.pytorch.load_model(saved, weights_only=True)
    finally:
        BentoMLContainer.model_store.reset()
