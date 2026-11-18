import json
import typing as t

import pytest
from pydantic import BaseModel

import bentoml
from _bentoml_sdk.service.openapi import generate_spec
from bentoml._internal.utils.cattr import bentoml_cattr


class Item(BaseModel):
    a: int
    b: str


class Other(BaseModel):
    c: float


def _collect_dangling_refs(svc: t.Any) -> t.List[str]:
    """Return the names of every ``#/components/schemas`` reference used in the
    generated paths that is *not* present under ``components/schemas``."""
    spec = json.loads(json.dumps(bentoml_cattr.unstructure(generate_spec(svc))))
    components = set(spec.get("components", {}).get("schemas", {}))
    refs: t.Set[str] = set()

    def walk(node: t.Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "$ref" and isinstance(value, str):
                    refs.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(spec.get("paths", {}))
    return sorted(
        name
        for ref in refs
        if ref.startswith("#/components/schemas/")
        and (name := ref.rsplit("/", 1)[-1]) not in components
    )


def _service_returning(annotation: t.Any) -> t.Any:
    def predict(self: t.Any, x: int = 0) -> t.Any: ...

    # Set the return annotation explicitly so the test keeps working regardless
    # of ``from __future__ import annotations`` semantics.
    predict.__annotations__["return"] = annotation

    namespace = {"predict": bentoml.api(predict)}
    return bentoml.service(type("Svc", (), namespace))


@pytest.mark.parametrize(
    "annotation",
    [
        t.Optional[Item],
        t.Union[Item, None],
        t.Union[Item, Other],
        t.List[Item],
        Item,
    ],
)
def test_openapi_spec_has_no_dangling_refs(annotation: t.Any) -> None:
    # Regression test for OpenAPI generation crashing / emitting dangling
    # references when an endpoint returns a model wrapped in Optional/Union.
    svc = _service_returning(annotation)
    assert _collect_dangling_refs(svc) == []


def test_optional_model_output_references_registered_component() -> None:
    svc = _service_returning(t.Optional[Item])
    spec = json.loads(json.dumps(bentoml_cattr.unstructure(generate_spec(svc))))

    schema = spec["paths"]["/predict"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert schema == {"$ref": "#/components/schemas/Item"}
    assert "Item" in spec["components"]["schemas"]
