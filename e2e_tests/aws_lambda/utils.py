import logging
import subprocess

import requests

logger = logging.getLogger('bentoml.test')


def send_test_data_to_endpoint(deployment_endpoint, sample_data=None):
    logger.info('Test deployment with sample request')
    sample_data = sample_data or '"{}"'
    request_result = requests.post(
        deployment_endpoint,
        data=sample_data,
        headers={'Content-Type': 'application/json'},
    )
    return request_result.status_code, request_result.content.decode('utf-8')


def run_lambda_create_or_update_command(deploy_command):
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

    if deploy_command_stdout.startswith(
        'Failed to create deployment'
    ) or deploy_command_stdout.startswith('Failed to update deployment'):
        return False, None

    deploy_command_stdout_list = deploy_command_stdout.split('\n')
    for index, message in enumerate(deploy_command_stdout_list):
        if '"endpoints": [' in message:
            deployment_endpoint = (
                deploy_command_stdout_list[index + 1].strip().replace('"', '')
            )
            return True, deployment_endpoint
    return False, None
