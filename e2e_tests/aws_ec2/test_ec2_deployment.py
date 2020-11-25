import logging
import uuid
import json

from sklearn import datasets

from bentoml.yatai.deployment.aws_utils import get_default_aws_region
from e2e_tests.cli_operations import delete_deployment
from e2e_tests.aws_ec2.utils import (
    run_aws_ec2_create_command,
    wait_for_healthy_targets_from_stack,
    send_test_data_to_multiple_endpoint,
)

logger = logging.getLogger('bentoml.test')


def test_aws_ec2_deployment(iris_clf_service):
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-ec2-e2e-{random_hash}'
    deployment_namespace = "dev"
    deployment_region = get_default_aws_region()

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
        deployment_endpoint = run_aws_ec2_create_command(create_deployment_command)

        instance_addresses = wait_for_healthy_targets_from_stack(
            name=deployment_name,
            namespace=deployment_namespace,
            region=deployment_region,
        )
        assert deployment_endpoint, "AWS EC2 deployment creation should success"
        assert instance_addresses, "AWS EC2 deployment should have all targets healthy"

        iris = datasets.load_iris()
        sample_data = iris.data[0:1]
        results = send_test_data_to_multiple_endpoint(
            [deployment_endpoint] + instance_addresses, json.dumps(sample_data.tolist())
        )
        for result in results:
            assert result[0] == 200, "prediction request should success"
            assert result[1] == '[0]', "prediction result mismatch"
    finally:
        delete_deployment('ec2', deployment_name, deployment_namespace)
