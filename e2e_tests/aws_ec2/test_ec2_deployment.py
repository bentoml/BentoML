import logging
import uuid
import json

from sklearn import datasets

from e2e_tests.cli_operations import delete_deployment
from e2e_tests.aws_ec2.utils import (
    run_aws_ec2_create_command,
    wait_for_instance_spawn,
    send_test_data_to_endpoint,
)

logger = logging.getLogger('bentoml.test')


def test_aws_ec2_deployment(iris_clf_service):
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-ec2-e2e-{random_hash}'
    deployment_namespace = "test"
    deployment_region = "ap-south-1"

    create_deployment_command = [
        'bentoml',
        'ec2',
        'deploy',
        deployment_name,
        '-b',
        iris_clf_service,
        '--namespace',
        deployment_namespace,
        '--region',
        deployment_region,
        '--verbose',
    ]

    try:
        deployment_success = run_aws_ec2_create_command(create_deployment_command)

        spawned, deployment_endpoints = wait_for_instance_spawn(
            name=deployment_name,
            namespace=deployment_namespace,
            region=deployment_region,
        )

        assert spawned, "AWS EC2 deployment should create ec2 instances"
        assert deployment_endpoints, "AWS EC2 deployment should have endpoint"

        assert deployment_success, "AWS Lambda deployment creation should success"
        assert deployment_endpoints, "AWS Lambda deployment should have endpoint"

        iris = datasets.load_iris()
        sample_data = iris.data[0:1]
        results = send_test_data_to_endpoint(
            deployment_endpoints, json.dumps(sample_data.tolist())
        )
        for result in results:
            assert result[0] == 200, "prediction request should success"
            assert result[1] == '[0]', "prediction result mismatch"
    finally:
        delete_deployment('ec2', deployment_name)
