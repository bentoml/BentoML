from __future__ import annotations

import json
from typing import Annotated
from typing import cast

import httpx
import numpy as np
import pytest
import torch
from pydantic import TypeAdapter
from starlette.applications import Starlette

import bentoml
from _bentoml_impl.serde import JSONSerde
from _bentoml_impl.serde import PickleSerde
from _bentoml_sdk.validators import TensorSchema
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("mode", ["json", "python"])
def test_tensor_schema_serialization(
    dtype: torch.dtype, requires_grad: bool, mode: str
):
    source = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=dtype, requires_grad=requires_grad
    )
    tensor = (source * 2).T
    adapter: TypeAdapter[torch.Tensor] = TypeAdapter(
        Annotated[torch.Tensor, TensorSchema("torch-tensor")]
    )

    if mode == "json":
        assert json.loads(adapter.dump_json(tensor)) == [[2.0, 6.0], [4.0, 8.0]]
    else:
        result = adapter.dump_python(tensor)
        np.testing.assert_array_equal(result, [[2.0, 6.0], [4.0, 8.0]])
        assert result.dtype == tensor.detach().numpy().dtype
        assert np.shares_memory(result, tensor.detach().numpy())
    assert tensor.requires_grad == requires_grad
    if requires_grad:
        tensor.sum().backward()
        torch.testing.assert_close(source.grad, torch.full_like(source, 2))


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("serde_type", [JSONSerde, PickleSerde])
def test_tensor_serde_serialization(
    requires_grad: bool, serde_type: type[JSONSerde] | type[PickleSerde]
):
    tensor = torch.tensor([1.0, 2.0], requires_grad=requires_grad) * 2
    schema = {"type": "tensor", "format": "torch-tensor"}
    serde = serde_type()

    payload = serde.serialize(tensor, schema)
    restored = serde.deserialize(payload, schema)

    torch.testing.assert_close(restored, tensor.detach())
    assert tensor.requires_grad == requires_grad


@pytest.mark.asyncio
async def test_torch_tensor_output_with_grad():
    @bentoml.service(metrics={"enabled": False})
    class TensorService:
        @bentoml.api
        def predict(self, values: list[float]) -> torch.Tensor:
            return torch.tensor(values, requires_grad=True) * 2

    app = cast(Starlette, TensorService.to_asgi())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/predict", json={"values": [1.0, 2.0]})
    assert response.status_code == 200
    assert response.json() == [2.0, 4.0]
