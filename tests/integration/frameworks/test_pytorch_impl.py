import torch
import pytest
from bentoml._internal.runner.container import AutoContainer
from bentoml._internal.frameworks.pytorch import PyTorchTensorContainer


# TODO: signatures
# TODO: to_payload with plasma

"""
@pytest.mark.parametrize(
    "bias_pair",
    [(0.0, 1.0), (-0.212, 1.1392)],
)
def test_pytorch_runner_with_partial_kwargs(bias_pair):

    N, D_in, H, D_out = 64, 1000, 100, 1
    x = torch.randn(N, D_in)
    model = ExtendedModel(D_in, H, D_out)

    tag = bentoml.pytorch.save_model("pytorch_test_extended", model)
    bias1, bias2 = bias_pair
    runner1 = bentoml.pytorch.load_runner(tag, partial_kwargs=dict(bias=bias1))

    runner2 = bentoml.pytorch.load_runner(tag, partial_kwargs=dict(bias=bias2))

    res1 = runner1.run_batch(x)[0][0].item()
    res2 = runner2.run_batch(x)[0][0].item()

    # tensor to float may introduce larger errors, so we bump rel_tol
    # from 1e-9 to 1e-6 just in case
    assert math.isclose(res1 - res2, bias1 - bias2, rel_tol=1e-6)
"""


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
