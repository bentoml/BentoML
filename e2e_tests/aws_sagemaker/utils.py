import logging
import subprocess

import boto3

from e2e_tests.yatai_server.utils import modified_environ

logger = logging.getLogger('bentoml.test')


def run_sagemaker_create_or_update_command(deploy_command):
    """
    :return: deployment_success, endpoint_name
    """
    logger.info(f"Running bentoml deploy command: {' '.join(deploy_command)}")
    with subprocess.Popen(
        deploy_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        deployment_stdout = proc.stdout.read().decode('utf-8')
    logger.info('Finish deploying to AWS Sagemaker')
    logger.info(deployment_stdout)

    if proc.returncode != 0:
        return False, None
    deployment_stdout_list = deployment_stdout.split('\n')
    for _, message in enumerate(deployment_stdout_list):
        if '"EndpointName":' in message:
            endpoint_name = message.split(':')[1].strip(',').replace('"', '')
            return True, endpoint_name
    return False, None


def send_test_data_to_endpoint(endpoint_name, sample_data=None, region="us-west-2"):
    logger.info(f'Test deployment with sample request for {endpoint_name}')
    sample_data = sample_data or '"[0]"'
    client = boto3.client('sagemaker-runtime')
    with modified_environ(AWS_REGION=region):
        result = client.invoke_endpoint(
            EndpointName=endpoint_name.strip(),
            ContentType='application/json',
            Body=sample_data,
        )
        logger.info(result)
        if result.get('ResponseMetadata', None) is None:
            return False, None
        if result['ResponseMetadata']['HTTPStatusCode'] != 200:
            return False, None
        body = result['Body'].read()
        return True, body.decode('utf-8')
