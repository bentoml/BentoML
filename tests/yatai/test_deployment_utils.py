import pytest

from bentoml.exceptions import YataiDeploymentException, BentoMLException
from bentoml.proto.deployment_pb2 import Deployment
from bentoml.yatai.deployment_utils import deployment_dict_to_pb


def test_deployment_dict_to_pb():
    failed_dict_no_operator = {'name': 'fake name'}
    with pytest.raises(YataiDeploymentException) as error:
        deployment_dict_to_pb(failed_dict_no_operator)
    assert str(error.value).startswith('"spec" is required field for deployment')

    failed_dict_custom_operator = {'name': 'fake', 'spec': {'operator': 'custom'}}
    with pytest.raises(BentoMLException) as error:
        deployment_dict_to_pb(failed_dict_custom_operator)
    assert str(error.value).startswith('Platform "custom" is not supported')

    deployment_dict = {
        'name': 'fake',
        'spec': {
            'operator': 'aws-lambda',
            'aws_lambda_operator_config': {'region': 'us-west-2'},
        },
    }
    result_pb = deployment_dict_to_pb(deployment_dict)
    assert isinstance(result_pb, Deployment)
    assert result_pb.name == 'fake'
