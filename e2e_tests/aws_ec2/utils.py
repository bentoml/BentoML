import boto3
import logging
import requests
from time import sleep

import subprocess
from bentoml.yatai.deployment.aws_utils import get_instance_ip_from_scaling_group
from bentoml.yatai.deployment.aws_ec2.utils import get_endpoints_from_instance_address

logger = logging.getLogger('bentoml.test')


def wait_for_instance_spawn(name, namespace, region):
    max_spawn_wait_retry = 10
    cf_client = boto3.client("cloudformation")

    # from stack get scaling group name
    stack_name = f"btml-stack-{namespace}-{name}".format(namespace=namespace, name=name)

    cloudformation_stack_result = cf_client.describe_stacks(StackName=stack_name)

    stack_result = cloudformation_stack_result.get("Stacks")[0]
    outputs = stack_result.get("Outputs")
    if not outputs:
        return None
    outputs = {o["OutputKey"]: o["OutputValue"] for o in outputs}

    autoscaling_group = outputs.get("AutoScalingGroup", None)
    if not autoscaling_group:
        return None

    while max_spawn_wait_retry > 0:
        addresses = get_instance_ip_from_scaling_group([autoscaling_group], region)
        endpoints = get_endpoints_from_instance_address(addresses, ["predict"])
        if endpoints:
            return endpoints
        max_spawn_wait_retry -= 1
        sleep(10)

    return None


def run_aws_ec2_create_command(deploy_command):
    """
    :return: deployment_success, deployment_endpoint
    """
    logger.info(f"Running bentoml deploy command: {' '.join(deploy_command)}")
    with subprocess.Popen(
        deploy_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        deploy_command_stdout = proc.stdout.read().decode('utf-8')

    logger.info('Finish deploying to AWS Lambda',)

    if proc.returncode != 0:
        return False, None

    deploy_command_stdout_list = deploy_command_stdout.split('\n')

    for index, message in enumerate(deploy_command_stdout_list):
        if '"Endpoints": [' in message:
            deployment_endpoint = (
                deploy_command_stdout_list[index + 1].strip().replace('"', '')
            )
            return True, deployment_endpoint
    return False, None


def send_test_data_to_multiple_endpoint(deployment_endpoints, sample_data=None):
    logger.info('Test deployment with sample request')
    sample_data = sample_data or '"{}"'
    all_results = []
    for endpoint in deployment_endpoints:
        request_result = requests.post(
            "http://" + endpoint,
            data=sample_data,
            headers={'Content-Type': 'application/json'},
        )
        all_results.append(
            (request_result.status_code, request_result.content.decode('utf-8'))
        )
    return all_results
