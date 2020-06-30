import logging
import uuid
import json

from sklearn import datasets

from e2e_tests.aws_lambda.utils import (
    send_test_data_to_endpoint,
    run_lambda_create_or_update_command,
)
from e2e_tests.cli_operations import delete_deployment

logger = logging.getLogger('bentoml.test')


def test_aws_lambda_deployment(iris_clf_service):
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-lambda-e2e-{random_hash}'

    create_deployment_command = [
        'bentoml',
        'lambda',
        'deploy',
        deployment_name,
        '-b',
        iris_clf_service,
        '--region',
        'us-west-2',
        '--verbose',
    ]
    try:
        deployment_success, deployment_endpoint = run_lambda_create_or_update_command(
            create_deployment_command
        )
        assert deployment_success, "AWS Lambda deployment creation should success"
        assert deployment_endpoint, "AWS Lambda deployment should have endpoint"

        iris = datasets.load_iris()
        sample_data = iris.data[0:1]
        status_code, content = send_test_data_to_endpoint(
            deployment_endpoint, json.dumps(sample_data.tolist())
        )
        assert status_code == 200, "prediction request should success"
        assert content == '[0]', "prediction result mismatch"
    finally:
        delete_deployment('lambda', deployment_name)
