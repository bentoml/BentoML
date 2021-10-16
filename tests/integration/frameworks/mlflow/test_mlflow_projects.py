import logging
import os
from pathlib import Path

import boto3
import botocore.exceptions
import pytest
from moto import mock_s3
from sklearn.neighbors import KNeighborsClassifier

import bentoml.mlflow

_exc_msg = """\
`boto3` is required to run tests.
 Install with `pip install boto3`
"""

current_file = Path(__file__).parent
MOCK_S3_BUCKET = "pytorch-mnist"
MOCK_S3_REGION = "us-east-1"


@pytest.fixture(scope="module", autouse=True)
def aws_cred():
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"


@pytest.fixture()
def mock_s3_client():
    with mock_s3():
        client = boto3.resource("s3", region_name=MOCK_S3_REGION).meta.client
        client.create_bucket(Bucket=MOCK_S3_BUCKET)
        return client


@pytest.fixture()
def s3_artifact_root(mock_s3_client):
    return f"s3://{MOCK_S3_BUCKET}"


@pytest.fixture()
def upload_files_to_s3(mock_s3_client, s3_artifact_root, aws_cred):
    try:
        with mock_s3():
            for f in Path(current_file, "SimpleMNIST").resolve().iterdir():
                _ = mock_s3_client.upload_file(
                    str(f), MOCK_S3_BUCKET, str(Path("SimpleMNIST", f.name))
                )
    except botocore.exceptions.ClientError as e:
        logging.error(e)
        return False
    return True


@pytest.mark.parametrize(
    "uri",
    [
        Path(current_file, "SimpleMNIST").resolve(),
        Path(current_file, "NestedMNIST").resolve(),
    ],
)
def test_mlflow_import_from_uri(uri, modelstore):
    tag = bentoml.mlflow.import_from_uri(str(uri), model_store=modelstore)
    model_info = modelstore.get(tag)
    assert "flavor" in model_info.options

    run, uri = bentoml.mlflow.load_project(tag, model_store=modelstore)
    assert callable(run)


def test_mlflow_import_from_uri_mlmodel(modelstore):
    uri = Path(current_file, "sklearn_clf").resolve()
    tag = bentoml.mlflow.import_from_uri(str(uri), model_store=modelstore)
    model_info = modelstore.get(tag)
    assert "flavor" in model_info.options
    model = bentoml.mlflow.load(tag, model_store=modelstore)
    assert isinstance(model, KNeighborsClassifier)


# TODO: tests with mock S3 buckets?
# def test_mlflow_import_from_s3_uri(upload_files_to_s3, s3_artifact_root, modelstore):
#     with mock_s3():
#         assert upload_files_to_s3
#         tag = bentoml.mlflow.import_from_uri(
#             s3_artifact_root, model_store=modelstore, s3_default_region=MOCK_S3_REGION
#         )
#         model_info = modelstore.get(tag)
#         assert "flavor" in model_info.options
