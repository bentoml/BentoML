import os

import pytest

import bentoml.mlflow


@pytest.fixture()
def envars():
    os.environ["PYTHONHASHSEED"] = "0"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Hello-t", "Hello_t"),
        ("0134-1j-ad", "bf8a75fd4f11c8bf69f598f8"),
        ("google/bert-L-1-W-128", "google_bert_L_1_W_128"),
    ],
)
def test_clean_name(name, expected, envars):
    assert bentoml.mlflow._clean_name(name) == expected


@pytest.mark.parametrize("uri, expected", [("s3:/test", True), ("https://not", False)])
def test_is_s3_url(uri, expected):
    assert bentoml.mlflow._is_s3_url(uri) == expected


@pytest.mark.parametrize("uri", ["/1234/4", "runs:/test-bucket", "s3://helloworld"])
def test_uri_to_filename(uri):
    assert bentoml.mlflow._uri_to_filename(uri).startswith("b")
