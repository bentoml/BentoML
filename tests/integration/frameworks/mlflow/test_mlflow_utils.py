import pytest

import bentoml.mlflow


@pytest.mark.parametrize("uri, expected", [("s3:/test", True), ("https://not", False)])
def test_is_s3_url(uri, expected):
    assert bentoml.mlflow._is_s3_url(uri) == expected


@pytest.mark.parametrize("uri", ["/1234/4", "runs:/test-bucket", "s3://helloworld"])
def test_uri_to_filename(uri):
    assert bentoml.mlflow._uri_to_filename(uri).startswith("mlf")
