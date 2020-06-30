import logging
import uuid

import json
from sklearn import datasets

from e2e_tests.aws_sagemaker.utils import (
    run_sagemaker_create_or_update_command,
    send_test_data_to_endpoint,
)
from e2e_tests.cli_operations import delete_deployment

logger = logging.getLogger('bentoml.test')


def test_sagemaker_deployment(iris_clf_service):

    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-sagemaker-e2e-{random_hash}'
    region = 'us-west-2'
    create_deployment_command = [
        'bentoml',
        'sagemaker',
        'deploy',
        deployment_name,
        '-b',
        iris_clf_service,
        '--region',
        region,
        '--api-name',
        'predict',
        '--num-of-gunicorn-workers-per-instance',
        '2',
        '--wait',
        '--verbose',
    ]

    try:
        deployment_success, endpoint_name = run_sagemaker_create_or_update_command(
            create_deployment_command
        )
        assert deployment_success, 'Sagemaker deployment was unsuccessful'
        assert endpoint_name, 'Sagemaker deployment endpoint name is missing'

        iris = datasets.load_iris()
        sample_data = iris.data[0:1]
        request_success, prediction_result = send_test_data_to_endpoint(
            endpoint_name, f'"{json.dumps(sample_data.tolist())}"', region
        )
        assert request_success, 'Failed to make successful Sagemaker prediction'
        assert (
            '[\n  0\n]\n' == prediction_result
        ), 'Sagemaker prediction result mismatches with expected value'
    finally:
        delete_deployment('sagemaker', deployment_name)
