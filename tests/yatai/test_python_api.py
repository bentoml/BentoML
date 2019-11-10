from mock import patch, Mock

from bentoml.proto.deployment_pb2 import (
    ApplyDeploymentResponse,
    DeleteDeploymentResponse,
    DescribeDeploymentResponse,
    GetDeploymentResponse, ListDeploymentsResponse)
from bentoml.proto.status_pb2 import Status
from bentoml.yatai.python_api import apply_deployment, create_deployment


def create_yatai_service_mock():
    yatai_service_mock = Mock()
    yatai_service_mock.ApplyDeployment.return_value = ApplyDeploymentResponse()
    yatai_service_mock.DeleteDeployment.return_value = DeleteDeploymentResponse()
    yatai_service_mock.DescribeDeployment.return_value = DescribeDeploymentResponse()
    yatai_service_mock.GetDeployment.return_value = GetDeploymentResponse()
    yatai_service_mock.ListDeployments.return_value = ListDeploymentsResponse()
    return yatai_service_mock


def test_apply_deployment_invalid_deployment_dict():
    yatai_service_mock = create_yatai_service_mock()
    invalid_deployment_yaml = {'spec': {'operator': 'aws-sagemaker'}}

    invalid_result = apply_deployment(invalid_deployment_yaml, yatai_service_mock)
    assert invalid_result.status.status_code == Status.ABORTED


def test_apply_deployment_successful():
    yatai_service_mock = create_yatai_service_mock()
    valid_deployment_yaml = {
        'name': 'deployment',
        'namespace': 'name',
        'spec': {
            'bento_name': 'ff',
            'bento_version': 'ff',
            'operator': 'aws-sagemaker',
            'sagemaker_operator_config': {}
        }
    }
    valid_result = apply_deployment(valid_deployment_yaml, yatai_service_mock)
    assert valid_result.status.status_code == Status.OK


# def test_create_deployment():
#     yatai_service_mock = create_yatai_service_mock()
#
#     invalid_operator_spec = {}
#     invalid_result = create_deployment()
#     assert True


