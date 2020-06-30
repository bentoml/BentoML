import logging
import subprocess
import time
import uuid

import requests

from e2e_tests.cli_operations import delete_deployment

logger = logging.getLogger('bentoml.test')


def test_azure_function_deployment(iris_clf_service):
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'test-azures-{random_hash}'
    command = f"""\
bentoml azure-functions deploy {deployment_name} -b {iris_clf_service} \
--location westus --max-burst 2 --function-auth-level anonymous --debug\
""".split(
        ' '
    )
    try:
        logger.info(f'Deploying {deployment_name} to Azure function')
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ) as proc:
            stdout = proc.stdout.read().decode('utf-8')
        logger.info(stdout)
        assert proc.returncode == 0, 'Failed to create Azure Functions deployment'
        deploy_command_stdout_list = stdout.split('\n')
        endpoint = None
        for index, message in enumerate(deploy_command_stdout_list):
            if '"hostNames": [' in message:
                endpoint = (
                    deploy_command_stdout_list[index + 1].strip().replace('"', '')
                )
        # Azure takes long time to download docker image, waiting at least 5 minutes
        # for it to be ready.
        logger.info('Sleeping 5 mins to wait for Azure download docker image')
        time.sleep(500)
        start_time = time.time()
        while (time.time() - start_time) < 400:
            logger.info(f'Making request to endpoint {endpoint}')
            request_result = requests.post(
                f'https://{endpoint}/predict',
                data='[[5, 4, 3, 2]]',
                headers={'Content-Type': 'application/json'},
            )
            logger.info(
                f'Request result {request_result.status_code} '
                f'{request_result.content.decode("utf-8")}'
            )
            if request_result.status_code == 503 or request_result.status_code == 502:
                time.sleep(100)
            else:
                break
        assert (
            request_result.status_code == 200
        ), 'Azure function deployment prediction request failed'
        assert (
            request_result.content.decode('utf-8') == '[1]'
        ), 'Azure function deployment prediction result mismatch'
    finally:
        delete_deployment('azure-functions', deployment_name)
