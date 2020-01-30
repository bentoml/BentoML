#!/usr/bin/env python

import subprocess
import logging
import uuid
import sys

import requests

from bentoml import BentoService, load, api
from bentoml.handlers import JsonHandler


logger = logging.getLogger('bentoml.test')


class TestLambdaDeployment(BentoService):
    @api(JsonHandler)
    def predict(self, data):
        return 'cat'


class UpdatedLambdaDeployment(BentoService):
    @api(JsonHandler)
    def predict(self, data):
        return 'dog'


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


def delete_deployment(deployment_name):
    logger.info('Delete test deployment with BentoML CLI')
    delete_deployment_command = [
        'bentoml',
        'lambda',
        'delete',
        deployment_name,
        '--force',
    ]
    logger.info(f'Delete command: {delete_deployment_command}')
    with subprocess.Popen(
        delete_deployment_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        delete_deployment_stdout = proc.stdout.read().decode('utf-8')
    logger.info(delete_deployment_stdout)


if __name__ == '__main__':
    deployment_failed = False
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-lambda-update-{random_hash}'

    args = sys.argv
    bento_name = None
    service_one = TestLambdaDeployment()
    saved_path = service_one.save()
    loaded_service = load(saved_path)
    bento_name = f'{loaded_service.name}:{loaded_service.version}'

    create_deployment_command = [
        'bentoml',
        'lambda',
        'deploy',
        deployment_name,
        '-b',
        bento_name,
        '--region',
        'us-west-2',
        '--verbose',
    ]
    deployment_failed, deployment_endpoint = run_lambda_create_or_update_command(
        create_deployment_command
    )

    if not deployment_failed and deployment_endpoint:
        deployment_failed = test_deployment_with_sample_data(
            deployment_endpoint, '"cat"'
        )

    service_two = UpdatedLambdaDeployment()
    service_two.save()
    updated_bento_name = f'{service_two.name}:{service_two.version}'
    update_deployment_command = [
        'bentoml',
        'lambda',
        'update',
        deployment_name,
        '-b',
        updated_bento_name,
        '--verbose',
    ]
    deployment_failed, deployment_endpoint = run_lambda_create_or_update_command(
        update_deployment_command
    )
    if not deployment_failed and deployment_endpoint:
        deployment_failed = test_deployment_with_sample_data(
            deployment_endpoint, '"dog"'
        )
    else:
        deployment_failed = True
        logger.debug('Update Lambda failed')

    delete_deployment(deployment_name)

    logger.info(f'Deleting bento service {bento_name}')
    delete_first_bento_command = ['bentoml', 'delete', bento_name, '-y']
    with subprocess.Popen(
        delete_first_bento_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        delete_first_bento_stdout = proc.stdout.read().decode('utf-8')
    logger.info(delete_first_bento_stdout)

    logger.info(f'Deleting bento service {updated_bento_name}')
    delete_updated_bento_command = ['bentoml', 'delete', updated_bento_name, '-y']
    with subprocess.Popen(
        delete_updated_bento_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        delete_updated_bento_stdout = proc.stdout.read().decode('utf-8')
    logger.info(delete_updated_bento_stdout)

    logger.info('Finished')
    if deployment_failed:
        logger.info('E2E deployment failed, fix the issues before releasing')
    else:
        logger.info('E2E Lambda deployment testing is successful')
