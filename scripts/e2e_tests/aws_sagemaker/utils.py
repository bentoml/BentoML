import logging
import subprocess

logger = logging.getLogger('bentoml.test')


def run_sagemaker_create_or_update_command(deploy_command):
    deployment_failed = False
    endpoint_name = ''
    logger.info(f"Running bentoml deploy command: {' '.join(deploy_command)}")
    with subprocess.Popen(
        deploy_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        deployment_stdout = proc.stdout.read().decode('utf-8')
    logger.info('Finish deploying to AWS Sagemaker')
    logger.info(deployment_stdout)
    # TODO
    if deployment_stdout.startswith('Failed to create deployment'):
        deployment_failed = True
    deployment_stdout_list = deployment_stdout.split('\n')
    for index, message in enumerate(deployment_stdout_list):
        if '"EndpointName":' in message:
            endpoint_name = message.split(':')[1].strip(',').replace('"', '')

    return deployment_failed, endpoint_name


def test_deployment_result(endpoint_name, expect_result, sample_data=None):
    logger.info(f'Test deployment with sample request for {endpoint_name}')
    deployment_failed = False
    sample_data = sample_data or '""'
    try:
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
            executable='/bin/bash',
        )
        logger.info(result)
        if result.stderr.decode('utf-8'):
            logger.error(result.stderr.decode('utf-8'))
            deployment_failed = True
        else:
            logger.info('Prediction Result: %s', result.stdout.decode('utf-8'))
            if expect_result == result.stdout.decode('utf-8'):
                deployment_failed = False
            else:
                deployment_failed = True
    except Exception as e:
        logger.error(str(e))
        deployment_failed = True

    return deployment_failed
