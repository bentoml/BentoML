import logging
import subprocess

import requests

logger = logging.getLogger('bentoml.test')


def test_deployment_with_sample_data(
    deployment_endpoint, expect_result, sample_data=None
):
    logger.info('Test deployment with sample request')
    sample_data = sample_data or '"{}"'
    deployment_failed = False
    try:
        request_result = requests.post(
            deployment_endpoint,
            data=sample_data,
            headers={'Content-Type': 'application/json'},
        )
        if request_result.status_code != 200:
            deployment_failed = True
        if request_result.content.decode('utf-8') != expect_result:
            logger.info(
                'Test request failed. {}:{}'.format(
                    request_result.status_code, request_result.content.decode('utf-8'),
                )
            )
            deployment_failed = True
    except Exception as e:
        logger.error(str(e))
        deployment_failed = True
    return deployment_failed


def run_lambda_create_or_update_command(deploy_command):
    logger.info(f"Running bentoml deploy command: {' '.join(deploy_command)}")
    deployment_failed = False
    deployment_endpoint = ''

    with subprocess.Popen(
        deploy_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        deploy_command_stdout = proc.stdout.read().decode('utf-8')
    logger.info('Finish deploying to AWS Lambda')
    logger.info(deploy_command_stdout)
    if deploy_command_stdout.startswith(
        'Failed to create deployment'
    ) or deploy_command_stdout.startswith('Failed to update deployment'):
        deployment_failed = True
    deploy_command_stdout_list = deploy_command_stdout.split('\n')
    for index, message in enumerate(deploy_command_stdout_list):
        if '"endpoints": [' in message:
            deployment_endpoint = (
                deploy_command_stdout_list[index + 1].strip().replace('"', '')
            )
    return deployment_failed, deployment_endpoint
