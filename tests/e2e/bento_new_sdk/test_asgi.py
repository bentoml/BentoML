from __future__ import annotations

from starlette.testclient import TestClient

import bentoml


@bentoml.service(metrics={"enabled": False})
class Model:
    model = {"hello": "world"}

    @bentoml.api()
    def get(self, key: str) -> str:
        return self.model.get(key, "")


@bentoml.service(metrics={"enabled": False})
class Service:
    model = bentoml.depends(Model)

    @bentoml.api()
    def get(self, key: str) -> dict[str, str]:
        return {"value": self.model.get(key)}


def test_simple_service():
    with TestClient(app=Model.to_asgi()) as client:
        resp = client.post("/get", json={"key": "hello"})

        assert resp.status_code == 200
        assert resp.text == "world"


def test_composed_service():
    with TestClient(app=Service.to_asgi()) as client:
        resp = client.post("/get", json={"key": "hello"})

        assert resp.status_code == 200
        assert resp.json() == {"value": "world"}
