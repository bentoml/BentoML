from bentoml.utils.validator import validate_deployment_pb_schema
from bentoml.proto.deployment_pb2 import Deployment, DeploymentOperator


def test_validate_deployment_pb_schema():
    test_pb = Deployment(name='test_deployment_name', namespace='namespace')
    test_pb.spec.bento_name = 'bento_name'
    test_pb.spec.bento_version = 'bento_version'
    test_pb.spec.operator = DeploymentOperator.Value('AWS_SAGEMAKER')
    test_pb.spec.sagemaker_operator_config.api_name = 'api_name'

    result = validate_deployment_pb_schema(test_pb)

    assert result is None

    test_bad_pb = test_pb
    test_bad_pb.name = ''
    bad_result = validate_deployment_pb_schema(test_bad_pb)
    assert bad_result == {'name': ['required field']}
