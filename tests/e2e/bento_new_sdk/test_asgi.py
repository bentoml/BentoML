from __future__ import annotations

from pathlib import Path

import pydantic
from fastapi import FastAPI
from starlette.testclient import TestClient

import bentoml


@bentoml.service(metrics={"enabled": False})
class Model:
    model = {"hello": "world"}

    @bentoml.api()
    def get(self, key: str) -> str:
        return self.model.get(key, "")


@bentoml.service(metrics={"enabled": False})
class RootInput:
    @bentoml.api()
    def get(self, key: str, /) -> str:
        return f"root-{key}"

    @bentoml.api()
    def read(self, path: Path, /) -> str:
        return path.read_text()

    @bentoml.api
    def chat(self, messages: list[str], /) -> str:
        return " ".join(messages)


class InputModel(pydantic.BaseModel):
    key: str


@bentoml.service(metrics={"enabled": False})
class RootModelInput:
    @bentoml.api
    def get(self, input: InputModel, /) -> str:
        return f"root-{input.key}"


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


def test_service_root_input(tmp_path: Path):
    with TestClient(app=RootInput.to_asgi()) as client:
        resp = client.post("/get", content="hello")

        assert resp.status_code == 200
        assert resp.text == "root-hello"

        tmp_path.joinpath("test.txt").write_text("hello")
        resp = client.post("/read", content=tmp_path.joinpath("test.txt").read_bytes())

        assert resp.status_code == 200
        assert resp.text == "hello"

        resp = client.post("/chat", json=["hello", "world"])

        assert resp.status_code == 200
        assert resp.text == "hello world"

    with TestClient(app=RootModelInput.to_asgi()) as client:
        resp = client.post("/get", json={"key": "hello"})

        assert resp.status_code == 200
        assert resp.text == "root-hello"


app = FastAPI()


@bentoml.service(path_prefix="/test-path")
@bentoml.asgi_app(app, path="/fastapi")
class PrefixedService:
    @bentoml.api()
    def ping(self) -> str:
        return "pong"

    @app.get("/greet/{name}")
    def greet(self, name: str):
        return {"message": f"hello {name}"}


def test_service_with_path_prefix():
    assert PrefixedService.path_prefix == "/test-path"
    assert PrefixedService.config["endpoints"]["livez"] == "/test-path/livez"

    with TestClient(app=PrefixedService.to_asgi(True)) as client:
        resp = client.get("/test-path")
        assert resp.status_code == 200

        resp = client.get("/test-path/livez")
        assert resp.status_code == 200

        resp = client.get("/test-path/metrics")
        assert resp.status_code == 200

        resp = client.get("/test-path/fastapi/greet/world")

        assert resp.status_code == 200
        assert resp.json() == {"message": "hello world"}

        resp = client.post("/test-path/ping")

        assert resp.status_code == 200
        assert resp.text == "pong"
