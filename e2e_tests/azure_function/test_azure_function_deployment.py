import logging
import subprocess
import time
import uuid

import requests
import json
from sklearn import datasets

from e2e_tests.cli_operations import delete_deployment

logger = logging.getLogger('bentoml.test')


def test_azure_function_deployment(iris_clf_service):
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-azure-function-{random_hash}'
    create_deployment_command = [
        'bentoml',
        'azure-function',
        'deploy',
        deployment_name,
        '-b',
        iris_clf_service,
        '--location',
        'westus',
        '--max-burst',
        '2',
        '--function-auth-level',
        'anonymous',
        '--debug',
    ]
    try:
        logger.info(f'Deploying {deployment_name} to Azure function')
        with subprocess.Popen(
            create_deployment_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as proc:
            stdout = proc.stdout.read().decode('utf-8')
        if stdout.startswith('Failed to create deployment'):
            raise Exception('Failed to create deployment')
        deploy_command_stdout_list = stdout.split('\n')
        endpoint=None
        for index, message in enumerate(deploy_command_stdout_list):
            if '"hostNames": [' in message:
                endpoint = (
                    deploy_command_stdout_list[index + 1].strip().replace('"', '')
                )
        iris = datasets.load_iris()
        sample_data = iris.data[0:1]
        start_time = time.time()
        while(time.time() - start_time) < 600:
            request_result = requests.post(
                f'https://{endpoint}/predict',
                data=f'"{json.dumps(sample_data.tolist())}"',
                headers={'Content-Type': 'application/json'},
            )
            if request_result.status_code == 502:
                time.sleep(60)
            else:
                break
        assert (
            request_result.status_code == 200
        ), 'Azure function deployment prediction request failed'
        assert (
            request_result.content.decode('utf-8') == '[0]'
        ), 'Azure function deployment prediction result mismatch'
    finally:
        delete_deployment('azure-function', deployment_name)
