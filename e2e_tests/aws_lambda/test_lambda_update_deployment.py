import uuid
import logging

from e2e_tests.aws_lambda.utils import (
    send_test_data_to_endpoint,
    run_lambda_create_or_update_command,
)
from e2e_tests.cli_operations import delete_deployment

logger = logging.getLogger('bentoml.test')


def test_aws_lambda_update_deployment(basic_bentoservice_v1, basic_bentoservice_v2):
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-lambda-update-{random_hash}'

    create_deployment_command = [
        'bentoml',
        'lambda',
        'deploy',
        deployment_name,
        '-b',
        basic_bentoservice_v1,
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
        status_code, content = send_test_data_to_endpoint(deployment_endpoint)
        assert status_code == 200, "prediction request should success"
        assert content == '"cat"', "prediction result mismatch"

        update_deployment_command = [
            'bentoml',
            'lambda',
            'update',
            deployment_name,
            '-b',
            basic_bentoservice_v2,
            '--verbose',
        ]

        (
            update_deployment_success,
            update_deployment_endpoint,
        ) = run_lambda_create_or_update_command(update_deployment_command)
        assert (
            update_deployment_success
        ), "AWS Lambda deployment creation should success"
        assert update_deployment_endpoint, "AWS Lambda deployment should have endpoint"

        status_code, content = send_test_data_to_endpoint(deployment_endpoint)
        assert status_code == 200, "Updated prediction request should success"
        assert content == '"dog"', "Updated prediction result mismatch"
    finally:
        delete_deployment('lambda', deployment_name)
