from bentoml.utils.validator import validate_deployment_pb_schema
from bentoml.proto.deployment_pb2 import Deployment, DeploymentSpec


def test_validate_deployment_pb_schema():
    test_pb = Deployment(name='test_deployment_name', namespace='namespace')
    test_pb.spec.bento_name = 'bento_name'
    test_pb.spec.bento_version = 'bento_version'
    test_pb.spec.operator = DeploymentSpec.DeploymentOperator.Value('AWS_SAGEMAKER')
    test_pb.spec.sagemaker_operator_config.api_name = 'api_name'
    test_pb.spec.sagemaker_operator_config.instance_type = 'mock_instance_type'
    test_pb.spec.sagemaker_operator_config.instance_count = 1

    result = validate_deployment_pb_schema(test_pb)

    assert result is None

    test_bad_pb = test_pb
    test_bad_pb.name = ''
    bad_result = validate_deployment_pb_schema(test_bad_pb)
    assert bad_result == {'name': ['required field']}


def test_validate_aws_lambda_schema():
    test_pb = Deployment(name='test_deployment_name', namespace='namespace')
    test_pb.spec.bento_name = 'bento_name'
    test_pb.spec.bento_version = 'bento_version'
    test_pb.spec.operator = DeploymentSpec.DeploymentOperator.Value('AWS_LAMBDA')
    test_pb.spec.aws_lambda_operator_config.api_name = 'api_name'
    test_pb.spec.aws_lambda_operator_config.region = 'us-west-2'
    test_pb.spec.aws_lambda_operator_config.timeout = 100
    test_pb.spec.aws_lambda_operator_config.memory_size = 128

    result = validate_deployment_pb_schema(test_pb)
    assert result is None

    test_pb.spec.aws_lambda_operator_config.timeout = 1000
    test_pb.spec.aws_lambda_operator_config.memory_size = 129
    failed_memory_test = validate_deployment_pb_schema(test_pb)
    print(failed_memory_test)
    aws_spec_fail_msg = failed_memory_test['spec'][0]['aws_lambda_operator_config'][0]

    assert aws_spec_fail_msg['memory_size']
    assert 'AWS Lambda memory' in aws_spec_fail_msg['memory_size'][0]
    assert aws_spec_fail_msg['timeout']
    assert 'max value is 900' in aws_spec_fail_msg['timeout'][0]
