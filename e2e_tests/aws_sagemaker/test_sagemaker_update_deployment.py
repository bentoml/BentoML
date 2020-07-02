import logging
import uuid

from e2e_tests.aws_sagemaker.utils import (
    run_sagemaker_create_or_update_command,
    send_test_data_to_endpoint,
)
from e2e_tests.cli_operations import delete_deployment


logger = logging.getLogger('bentoml.test')


def test_sagemaker_update_deployment(basic_bentoservice_v1, basic_bentoservice_v2):
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-sagemaker-update-e2e-{random_hash}'
    region = 'us-west-2'

    create_deployment_command = [
        'bentoml',
        'sagemaker',
        'deploy',
        deployment_name,
        '-b',
        basic_bentoservice_v1,
        '--api-name',
        'predict',
        '--region',
        region,
        '--verbose',
    ]
    try:
        deployment_success, endpoint_name = run_sagemaker_create_or_update_command(
            create_deployment_command
        )
        assert deployment_success, 'Sagemaker deployment was unsuccessful'
        assert endpoint_name, 'Sagemaker deployment endpoint name is missing'

        request_success, prediction_result = send_test_data_to_endpoint(
            endpoint_name, region=region
        )
        assert request_success, 'Failed to make successful Sagemaker request'
        assert (
            prediction_result.strip() == '"cat"'
        ), 'Sagemaker prediction result mismatch'

        update_bento_version_deployment_command = [
            'bentoml',
            'sagemaker',
            'update',
            deployment_name,
            '-b',
            basic_bentoservice_v2,
            '--wait',
            '--verbose',
        ]
        (
            updated_deployment_success,
            endpoint_name,
        ) = run_sagemaker_create_or_update_command(
            update_bento_version_deployment_command
        )
        assert (
            updated_deployment_success
        ), 'Sagemaker update deployment was unsuccessful'
        assert endpoint_name, 'Sagemaker deployment endpoint name is missing'

        request_success, prediction_result = send_test_data_to_endpoint(
            endpoint_name, region=region
        )
        assert request_success, 'Failed to make successful Sagemaker request'
        assert (
            prediction_result.strip() == '"dog"'
        ), 'Sagemaker prediction result mismatch'
    finally:
        delete_deployment('sagemaker', deployment_name)
