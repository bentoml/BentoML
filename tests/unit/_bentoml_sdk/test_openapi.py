from __future__ import annotations

import pydantic
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

import bentoml


class Prediction(pydantic.BaseModel):
    score: int

    @pydantic.computed_field
    @property
    def label(self) -> str:
        return "positive" if self.score > 0 else "negative"


class Predictions(pydantic.BaseModel):
    predictions: list[Prediction]


@bentoml.service(metrics={"enabled": False})
class PredictionService:
    @bentoml.api
    def predict(self, score: int) -> Predictions:
        return Predictions(predictions=[Prediction(score=score)])


def test_response_components_include_computed_fields():
    with TestClient(PredictionService.to_asgi()) as client:
        response = client.post("/predict", json={"score": 3})
        assert response.status_code == 200
        assert response.json() == {"predictions": [{"score": 3, "label": "positive"}]}
        schema_response = client.get("/docs.json")
        assert schema_response.status_code == 200
        schema = schema_response.json()

    prediction = schema["components"]["schemas"]["Prediction"]
    assert prediction["properties"]["label"]["type"] == "string"
    assert prediction["properties"]["label"]["readOnly"] is True
    assert "label" in prediction["required"]


class FormattedPrediction(pydantic.BaseModel):
    score: int

    @pydantic.field_serializer("score")
    def serialize_score(self, score: int) -> str:
        return f"{score}%"


class FormattedPredictions(pydantic.BaseModel):
    predictions: list[FormattedPrediction]


@bentoml.service(metrics={"enabled": False})
class FormattedPredictionService:
    @bentoml.api
    def predict(self, score: int) -> FormattedPredictions:
        return FormattedPredictions(predictions=[FormattedPrediction(score=score)])


def test_response_components_match_field_serializers():
    with TestClient(FormattedPredictionService.to_asgi()) as client:
        response = client.post("/predict", json={"score": 3})
        assert response.status_code == 200
        assert response.json() == {"predictions": [{"score": "3%"}]}
        schema_response = client.get("/docs.json")
        assert schema_response.status_code == 200
        schema = schema_response.json()

    prediction = schema["components"]["schemas"]["FormattedPrediction"]
    assert prediction["properties"]["score"]["type"] == "string"


class SharedPrediction(pydantic.BaseModel):
    score: int

    @pydantic.field_serializer("score")
    def serialize_score(self, score: int) -> str:
        return f"{score}%"

    @pydantic.computed_field
    @property
    def label(self) -> str:
        return "positive" if self.score > 0 else "negative"


class SharedPredictions(pydantic.BaseModel):
    prediction: SharedPrediction


@bentoml.service(metrics={"enabled": False})
class SharedPredictionService:
    @bentoml.api
    def predict(self, prediction: SharedPrediction) -> SharedPredictions:
        return SharedPredictions(prediction=prediction)

    @bentoml.api
    def repeat(self, prediction: SharedPrediction) -> SharedPredictions:
        return SharedPredictions(prediction=prediction)


@pytest.mark.parametrize("route", ["/predict", "/repeat"])
def test_shared_components_preserve_request_and_response_modes(route: str):
    with TestClient(SharedPredictionService.to_asgi()) as client:
        response = client.post(route, json={"prediction": {"score": 3}})
        assert response.status_code == 200
        assert response.json() == {"prediction": {"score": "3%", "label": "positive"}}
        invalid = client.post(route, json={"prediction": {"score": "3%"}})
        assert invalid.status_code == 400
        schema_response = client.get("/docs.json")
        assert schema_response.status_code == 200
        schema = schema_response.json()

    operation = schema["paths"][route]["post"]
    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    response_schema = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]
    input_ref = request_schema["properties"]["prediction"]["$ref"]
    output_ref = response_schema["properties"]["prediction"]["$ref"]
    components = schema["components"]["schemas"]
    input_prediction = components[input_ref.rsplit("/", 1)[1]]
    output_prediction = components[output_ref.rsplit("/", 1)[1]]
    assert input_prediction["properties"]["score"]["type"] == "integer"
    assert "label" not in input_prediction["properties"]
    assert output_prediction["properties"]["score"]["type"] == "string"
    assert output_prediction["properties"]["label"]["readOnly"] is True
    assert "label" in output_prediction["required"]
    assert input_ref != output_ref


@bentoml.service(metrics={"enabled": False})
class RootPredictionService:
    @bentoml.api
    def predict(self, scores: list[int], /) -> list[int]:
        return scores


def test_root_request_and_response_schemas_remain_unwrapped():
    with TestClient(RootPredictionService.to_asgi()) as client:
        response = client.post("/predict", json=[1, 2])
        assert response.status_code == 200
        assert response.json() == [1, 2]
        schema_response = client.get("/docs.json")
        assert schema_response.status_code == 200
        operation = schema_response.json()["paths"]["/predict"]["post"]

    request = operation["requestBody"]["content"]["application/json"]["schema"]
    response = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert request["type"] == "array"
    assert request["items"]["type"] == "integer"
    assert request["root_input"] is True
    assert response["type"] == "array"
    assert response["items"]["type"] == "integer"


class Input(pydantic.BaseModel):
    fastapi_value: str


mounted_app = FastAPI()


@mounted_app.post("/echo")
def mounted_echo(value: Input) -> Input:
    return value


@bentoml.service(metrics={"enabled": False})
@bentoml.asgi_app(mounted_app, path="/mounted")
class MountedPredictionService:
    @bentoml.api
    def predict(self, score: int) -> SharedPredictions:
        return SharedPredictions(prediction=SharedPrediction(score=score))


def test_mounted_application_components_are_not_overwritten():
    with TestClient(MountedPredictionService.to_asgi()) as client:
        response = client.post("/mounted/echo", json={"fastapi_value": "hello"})
        assert response.status_code == 200
        assert response.json() == {"fastapi_value": "hello"}
        response = client.post("/predict", json={"score": 3})
        assert response.status_code == 200
        assert response.json() == {"prediction": {"score": "3%", "label": "positive"}}
        schema_response = client.get("/docs.json")
        assert schema_response.status_code == 200
        schema = schema_response.json()

    mounted_operation = schema["paths"]["/mounted/echo"]["post"]
    request_schema = mounted_operation["requestBody"]["content"]["application/json"][
        "schema"
    ]
    component = schema["components"]["schemas"][
        request_schema["$ref"].rsplit("/", 1)[1]
    ]
    assert component["properties"]["fastapi_value"]["type"] == "string"
    operation = schema["paths"]["/predict"]["post"]
    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema["properties"]["score"]["type"] == "integer"
    response_schema = operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ]
    output_ref = response_schema["properties"]["prediction"]["$ref"]
    output = schema["components"]["schemas"][output_ref.rsplit("/", 1)[1]]
    assert output["properties"]["score"]["type"] == "string"
    assert output["properties"]["label"]["readOnly"] is True
