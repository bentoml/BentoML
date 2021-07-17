import pytest

from bentoml.exceptions import BentoMLException, YataiDeploymentException
from bentoml.yatai.deployment_utils import deployment_dict_to_pb
from bentoml.yatai.proto.deployment_pb2 import Deployment


def test_deployment_dict_to_pb():
    failed_dict_no_operator = {"name": "fake name"}
    with pytest.raises(YataiDeploymentException) as error:
        deployment_dict_to_pb(failed_dict_no_operator)
    assert str(error.value).startswith('"spec" is required field for deployment')

    failed_dict_custom_operator = {"name": "fake", "spec": {"operator": "custom"}}
    with pytest.raises(BentoMLException) as error:
        deployment_dict_to_pb(failed_dict_custom_operator)
    assert str(error.value).startswith('Platform "custom" is not supported')

    deployment_dict = {
        "name": "fake",
        "spec": {
            "operator": "aws-lambda",
            "aws_lambda_operator_config": {"region": "us-west-2"},
        },
    }
    result_pb = deployment_dict_to_pb(deployment_dict)
    assert isinstance(result_pb, Deployment)
    assert result_pb.name == "fake"


def test_deployment_dict_to_pb_for_lambda():
    deployment_dict = {
        "name": "mock",
        "spec": {
            "operator": "aws-lambda",
            "aws_lambda_operator_config": {
                "region": "us-west-2",
                "memory_size": 1024,
                "timeout": 60,
            },
        },
    }
    result_pb = deployment_dict_to_pb(deployment_dict)
    assert result_pb.spec.aws_lambda_operator_config.memory_size == 1024
    assert result_pb.spec.aws_lambda_operator_config.region == "us-west-2"


def test_deployment_dict_to_pb_for_sagemaker():
    deployment_dict = {
        "name": "mock",
        "spec": {
            "operator": "aws-sagemaker",
            "sagemaker_operator_config": {
                "region": "us-west-2",
                "api_name": "predict",
                "instance_type": "mock_type",
                "num_of_gunicorn_workers_per_instance": 4,
            },
        },
    }
    result_pb = deployment_dict_to_pb(deployment_dict)
    assert result_pb.spec.sagemaker_operator_config.region == "us-west-2"
    assert result_pb.spec.sagemaker_operator_config.api_name == "predict"
    assert result_pb.spec.sagemaker_operator_config.instance_type == "mock_type"
