from __future__ import annotations

import pydantic
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
