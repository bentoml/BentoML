from mock import Mock

from bentoml.proto.deployment_pb2 import (
    ApplyDeploymentResponse,
    DeleteDeploymentResponse,
    DescribeDeploymentResponse,
    GetDeploymentResponse,
    ListDeploymentsResponse,
)
from bentoml.proto import status_pb2
from bentoml.yatai.python_api import apply_deployment, create_deployment
from bentoml.yatai.status import Status


def create_yatai_service_mock():
    yatai_service_mock = Mock()
    yatai_service_mock.ApplyDeployment.return_value = ApplyDeploymentResponse()
    yatai_service_mock.DeleteDeployment.return_value = DeleteDeploymentResponse()
    yatai_service_mock.DescribeDeployment.return_value = DescribeDeploymentResponse()
    yatai_service_mock.GetDeployment.return_value = GetDeploymentResponse(
        status=Status.NOT_FOUND()
    )
    yatai_service_mock.ListDeployments.return_value = ListDeploymentsResponse()
    return yatai_service_mock


def test_apply_deployment_invalid_deployment_dict():
    yatai_service_mock = create_yatai_service_mock()
    invalid_deployment_yaml = {'spec': {'operator': 'aws-sagemaker'}}

    invalid_result = apply_deployment(invalid_deployment_yaml, yatai_service_mock)
    assert invalid_result.status.status_code == status_pb2.Status.INVALID_ARGUMENT


def test_apply_deployment_successful():
    yatai_service_mock = create_yatai_service_mock()
    valid_deployment_yaml = {
        'name': 'deployment',
        'namespace': 'name',
        'spec': {
            'bento_name': 'ff',
            'bento_version': 'ff',
            'operator': 'aws-sagemaker',
            'sagemaker_operator_config': {},
        },
    }
    valid_result = apply_deployment(valid_deployment_yaml, yatai_service_mock)
    assert valid_result.status.status_code == status_pb2.Status.OK


def test_create_deployment_failed_with_no_api_name():
    yatai_service_mock = create_yatai_service_mock()
    deployment_name = 'deployment'
    namespace = 'namespace'
    bento_name = 'name'
    bento_version = 'version'
    platform = 'aws-sagemaker'
    invalid_operator_spec = {}
    result = create_deployment(
        deployment_name=deployment_name,
        namespace=namespace,
        bento_name=bento_name,
        bento_version=bento_version,
        platform=platform,
        operator_spec=invalid_operator_spec,
        labels={},
        annotations={},
        yatai_service=yatai_service_mock,
    )
    assert result.status.status_code == status_pb2.Status.INVALID_ARGUMENT


def test_create_deployment_successful():
    yatai_service_mock = create_yatai_service_mock()
    deployment_name = 'deployment'
    namespace = 'namespace'
    bento_name = 'name'
    bento_version = 'version'
    platform = 'aws-sagemaker'
    operator_spec = {
        'api_name': 'predict', 'instance_type': 'mock_instance', 'instance_count': 1
    }
    result = create_deployment(
        deployment_name=deployment_name,
        namespace=namespace,
        bento_name=bento_name,
        bento_version=bento_version,
        platform=platform,
        operator_spec=operator_spec,
        labels={},
        annotations={},
        yatai_service=yatai_service_mock,
    )
    assert result.status.status_code == status_pb2.Status.OK
