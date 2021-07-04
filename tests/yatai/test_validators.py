from bentoml.yatai.validator import validate_deployment_pb
from bentoml.yatai.proto.deployment_pb2 import Deployment, DeploymentSpec


def _get_test_custom_deployment_pb():
    test_pb = Deployment(name='test_deployment_name', namespace='namespace')
    test_pb.spec.bento_name = 'bento_name'
    test_pb.spec.bento_version = 'bento_version'
    test_pb.spec.operator = DeploymentSpec.DeploymentOperator.Value('CUSTOM')
    test_pb.spec.sagemaker_operator_config.api_name = 'api_name'
    test_pb.spec.sagemaker_operator_config.instance_type = 'mock_instance_type'
    test_pb.spec.sagemaker_operator_config.instance_count = 1
    return test_pb


def test_validate_deployment_pb_schema():
    deployment_pb = _get_test_custom_deployment_pb()
    assert validate_deployment_pb(deployment_pb) is None

    deployment_pb_with_empty_name = _get_test_custom_deployment_pb()
    deployment_pb_with_empty_name.name = ''
    errors = validate_deployment_pb(deployment_pb_with_empty_name)
    assert errors == {'name': ['required field']}

    deployment_pb_with_invalid_service_version = _get_test_custom_deployment_pb()
    deployment_pb_with_invalid_service_version.spec.bento_version = 'latest'
    errors = validate_deployment_pb(deployment_pb_with_invalid_service_version)
    assert errors['spec'][0]['bento_version'] == [
        'Must use specific "bento_version" in deployment, using "latest" is an '
        'anti-pattern.'
    ]
