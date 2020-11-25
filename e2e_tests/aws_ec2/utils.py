import boto3
import logging
import requests
from time import sleep
import subprocess

logger = logging.getLogger('bentoml.test')


def get_instance_ip_from_id(instance_id, api, region):
    """get instance ip address from instance_id,
    and append bentoservice docker port and api_name"""

    ec2_client = boto3.client("ec2", region)
    response = ec2_client.describe_instances(InstanceIds=[instance_id])
    ip_address = response["Reservations"][0]["Instances"][0]["PublicIpAddress"]
    url = "http://" + ip_address + ":" + "5000" + api
    return url


def get_addresses_from_target_group(target_group_arn, api, region):
    """Get addresses of all healthy ec2 targets of target group.
    If any target is not healthy (health checks failing) return None"""

    healthy_target_status = "healthy"
    eb_client = boto3.client("elbv2", region)

    all_targets = eb_client.describe_target_health(TargetGroupArn=target_group_arn)
    all_addresses = []
    for instance in all_targets["TargetHealthDescriptions"]:
        health = instance["TargetHealth"]["State"]
        if not health == healthy_target_status:
            # if any target is unhealthy,wait.
            return None
        id_ = instance["Target"]["Id"]
        all_addresses.append(get_instance_ip_from_id(id_, api, region))
    return all_addresses


def wait_for_healthy_targets_from_stack(name, namespace, region):
    """From target group of stack,wait for all targets to be healthy.
    Then return their '/predict' api url"""

    max_spawn_wait_retry = 30
    cf_client = boto3.client("cloudformation", region)

    stack_name = f"btml-stack-{namespace}-{name}".format(namespace=namespace, name=name)
    cloudformation_stack_result = cf_client.describe_stacks(StackName=stack_name)
    logger.info('Describe cloud formation result')
    logger.info(cloudformation_stack_result)

    stack_result = cloudformation_stack_result.get("Stacks")[0]
    outputs = stack_result.get("Outputs")
    if not outputs:
        return None
    outputs = {o["OutputKey"]: o["OutputValue"] for o in outputs}

    target_group_arn = outputs.get("TargetGroup", None)
    if not target_group_arn:
        return None

    while max_spawn_wait_retry > 0:
        addresses = get_addresses_from_target_group(
            target_group_arn, "/predict", region
        )
        if addresses:
            return addresses
        max_spawn_wait_retry -= 1
        sleep(15)

    return None


def get_url_from_deploy_stdout(stdout):
    """Return ELB url from deployment stdout"""

    loadbalancer_endpoint = None
    for index, message in enumerate(stdout):
        if '"Url":' in message:
            loadbalancer_endpoint = (
                stdout[index].strip().replace('"Url": ', '').replace('"', '')
            )
    return loadbalancer_endpoint


def run_aws_ec2_create_command(deploy_command):
    """
    :return: deployment_success, deployment_endpoint
    """
    logger.info(f"Running bentoml deploy command: {' '.join(deploy_command)}")
    with subprocess.Popen(
        deploy_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        deploy_command_stdout = proc.stdout.read().decode('utf-8')
    logger.info('Finish deploying to AWS Lambda')
    logger.info(deploy_command_stdout)

    if proc.returncode != 0:
        return None
    deploy_command_stdout_list = deploy_command_stdout.split('\n')
    url = get_url_from_deploy_stdout(deploy_command_stdout_list)
    if url:
        return url + "/predict"
    return None


def send_test_data_to_multiple_endpoint(deployment_endpoints, sample_data=None):
    """Post to multiple endpoints and return aggregated results"""

    logger.info('Test deployment with sample request')
    sample_data = sample_data or '"{}"'
    all_results = []
    for endpoint in deployment_endpoints:
        request_result = requests.post(
            endpoint, data=sample_data, headers={'Content-Type': 'application/json'},
        )
        logger.info(f'Sending request to {endpoint}')
        logging.info(
            f'Request result: {request_result.status_code} '
            f'{request_result.content.decode("utf-8")}'
        )
        all_results.append(
            (request_result.status_code, request_result.content.decode('utf-8'))
        )
    return all_results
