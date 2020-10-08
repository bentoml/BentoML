from bentoml.yatai.validator import validate_deployment_pb
from bentoml.yatai.proto.deployment_pb2 import Deployment, DeploymentSpec


def _get_test_sagemaker_deployment_pb():
    test_pb = Deployment(name='test_deployment_name', namespace='namespace')
    test_pb.spec.bento_name = 'bento_name'
    test_pb.spec.bento_version = 'bento_version'
    test_pb.spec.operator = DeploymentSpec.DeploymentOperator.Value('AWS_SAGEMAKER')
    test_pb.spec.sagemaker_operator_config.api_name = 'api_name'
    test_pb.spec.sagemaker_operator_config.instance_type = 'mock_instance_type'
    test_pb.spec.sagemaker_operator_config.instance_count = 1
    return test_pb


def _get_test_lambda_deployment_pb():
    test_pb = Deployment(name='test_deployment_name', namespace='namespace')
    test_pb.spec.bento_name = 'bento_name'
    test_pb.spec.bento_version = 'bento_version'
    test_pb.spec.operator = DeploymentSpec.DeploymentOperator.Value('AWS_LAMBDA')
    test_pb.spec.aws_lambda_operator_config.api_name = 'api_name'
    test_pb.spec.aws_lambda_operator_config.region = 'us-west-2'
    test_pb.spec.aws_lambda_operator_config.timeout = 100
    test_pb.spec.aws_lambda_operator_config.memory_size = 128
    return test_pb


def test_validate_deployment_pb_schema():
    deployment_pb = _get_test_sagemaker_deployment_pb()
    assert validate_deployment_pb(deployment_pb) is None

    deployment_pb_with_empty_name = _get_test_sagemaker_deployment_pb()
    deployment_pb_with_empty_name.name = ''
    errors = validate_deployment_pb(deployment_pb_with_empty_name)
    assert errors == {'name': ['required field']}

    deployment_pb_with_invalid_service_version = _get_test_sagemaker_deployment_pb()
    deployment_pb_with_invalid_service_version.spec.bento_version = 'latest'
    errors = validate_deployment_pb(deployment_pb_with_invalid_service_version)
    assert errors['spec'][0]['bento_version'] == [
        'Must use specific "bento_version" in deployment, using "latest" is an '
        'anti-pattern.'
    ]


def test_validate_aws_lambda_schema():
    deployment_pb = _get_test_lambda_deployment_pb()
    assert validate_deployment_pb(deployment_pb) is None

    deployment_pb_with_bad_memory_size = _get_test_lambda_deployment_pb()
    deployment_pb_with_bad_memory_size.spec.aws_lambda_operator_config.timeout = 1000
    deployment_pb_with_bad_memory_size.spec.aws_lambda_operator_config.memory_size = 129
    errors = validate_deployment_pb(deployment_pb_with_bad_memory_size)
    aws_spec_fail_msg = errors['spec'][0]['aws_lambda_operator_config'][0]

    assert 'AWS Lambda memory' in aws_spec_fail_msg['memory_size'][0]
    assert 'max value is 900' in aws_spec_fail_msg['timeout'][0]
