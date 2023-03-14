from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import pytest

from bentoml.io import IODescriptor

if TYPE_CHECKING:
    from bentoml._internal.context import InferenceApiContext as Context


class DummyDescriptor(
    IODescriptor[t.Any],
    descriptor_id="bentoml.io.Dummy",
    proto_fields=("serialized_bytes",),
):
    _mime_type = "application/vnd.bentoml.dummy"

    def __init__(self, **kwargs: t.Any):
        [object.__setattr__(self, k, v) for k, v in kwargs.items()]

    def openapi_schema(self) -> t.Any:
        raise NotImplementedError

    def openapi_components(self) -> dict[str, t.Any] | None:
        raise NotImplementedError

    def openapi_example(self) -> t.Any | None:
        raise NotImplementedError

    def openapi_request_body(self) -> dict[str, t.Any]:
        raise NotImplementedError

    def openapi_responses(self) -> dict[str, t.Any]:
        raise NotImplementedError

    def to_spec(self) -> dict[str, t.Any]:
        raise NotImplementedError

    @classmethod
    def from_spec(cls, spec: dict[str, t.Any]) -> t.Self:
        return cls(**spec)

    def input_type(self) -> t.Any:
        return str

    async def from_http_request(self, request: t.Any) -> t.Any:
        return request

    async def to_http_response(self, obj: t.Any, ctx: Context | None = None) -> t.Any:
        return obj, ctx

    async def from_proto(self, field: t.Any) -> t.Any:
        return field

    async def to_proto(self, obj: t.Any) -> t.Any:
        return obj

    def _from_sample(self, sample: t.Any):
        return sample


@pytest.mark.parametrize(
    "fn",
    [
        f"openapi_{n}"
        for n in ("schema", "components", "example", "request_body", "responses")
    ],
)
def test_raise_not_implemented_openapi(fn: str):
    with pytest.raises(NotImplementedError):
        getattr(DummyDescriptor(), fn)()
