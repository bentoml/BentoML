import logging
import subprocess

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

    if deployment_stdout.startswith(
        'Failed to create AWS Sagemaker deployment'
    ) or deployment_stdout.startswith('Failed to update AWS Sagemaker deployment'):
        return False, None
    deployment_stdout_list = deployment_stdout.split('\n')
    for _, message in enumerate(deployment_stdout_list):
        if '"EndpointName":' in message:
            endpoint_name = message.split(':')[1].strip(',').replace('"', '')
            return True, endpoint_name
    return False, None


def send_test_data_to_endpoint(endpoint_name, sample_data=None):
    logger.info(f'Test deployment with sample request for {endpoint_name}')
    sample_data = sample_data or '"[0]"'
    test_command = [
        'aws',
        'sagemaker-runtime',
        'invoke-endpoint',
        '--endpoint-name',
        endpoint_name,
        '--content-type',
        '"application/json"',
        '--body',
        sample_data,
        '>(cat) 1>/dev/null',
        '|',
        'jq .',
    ]
    logger.info('Testing command: %s', ' '.join(test_command))
    result = subprocess.run(
        ' '.join(test_command),
        capture_output=True,
        shell=True,
        check=True,
        executable='/bin/bash',
    )
    logger.info(result)
    if result.stderr.decode('utf-8'):
        return False, None
    else:
        return True, result.stdout.decode('utf-8')
