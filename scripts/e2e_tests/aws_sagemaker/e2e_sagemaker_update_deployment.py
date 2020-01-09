#!/usr/bin/env python

import subprocess
import logging
import uuid

from bentoml import BentoService, load, api
from bentoml.handlers import DataframeHandler

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

    if deployment_stdout.startswith(
        'Failed to create deployment'
    ) or deployment_stdout.startswith('Failed to update deployment'):
        deployment_failed = True
        return deployment_failed, endpoint_name
    deployment_stdout_list = deployment_stdout.split('\n')
    for index, message in enumerate(deployment_stdout_list):
        if '"EndpointName":' in message:
            endpoint_name = message.split(':')[1].strip(',').replace('"', '')

    return deployment_failed, endpoint_name


def test_deployment_result(endpoint_name, expect_result, sample_data=None):
    logger.info(f'Test deployment with sample request for {endpoint_name}')
    deployment_failed = False
    sample_data = sample_data or '"[0]"'
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
            logger.info(
                f"Did deployment failed? {deployment_failed}  "
                f"Actual result '{result.stdout.decode('utf-8')}', and expect "
                f"result '{expect_result}'"
            )
    except Exception as e:
        logger.error(str(e))
        deployment_failed = True

    return deployment_failed


def delete_deployment(deployment_name):
    logger.info('Delete test deployment with BentoML CLI')
    delete_deployment_command = [
        'bentoml',
        'deploy',
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


class TestDeploymentService(BentoService):
    @api(DataframeHandler)
    def predict(self, df):
        return 1

    @api(DataframeHandler)
    def classify(self, df):
        return 'cat'


class UpdatedTestDeploymentService(BentoService):
    @api(DataframeHandler)
    def predict(self, df):
        return 1

    @api(DataframeHandler)
    def classify(self, df):
        # change result from cat to dog
        return 'dog'


if __name__ == '__main__':
    deployment_failed = False
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-sagemaker-update-e2e-{random_hash}'
    region = 'us-west-2'

    logger.info('Creating version one BentoService bundle..')
    service_ver_one = TestDeploymentService()
    saved_path = service_ver_one.save()

    loaded_ver_two_service = load(saved_path)
    bento_name = f'{loaded_ver_two_service.name}:{loaded_ver_two_service.version}'
    create_deployment_command = [
        'bentoml',
        '--verbose',
        'deploy',
        'create',
        deployment_name,
        '-b',
        bento_name,
        '--platform',
        'aws-sagemaker',
        '--api-name',
        'classify',
    ]
    deployment_failed, endpoint_name = run_sagemaker_create_or_update_command(
        create_deployment_command
    )

    if not deployment_failed and endpoint_name:
        deployment_failed = test_deployment_result(endpoint_name, '"cat"\n')
    else:
        deployment_failed = True
        logger.info('Deployment failed for creating deployment')

    # if not deployment_failed:
    #     logger.info('UPDATED ENV FOR DEPLOYMENT')
    #     update_deployment_command = [
    #         'bentoml',
    #         '--verbose',
    #         'deploy',
    #         'update',
    #         deployment_name,
    #         '--api-name',
    #         'predict',
    #     ]
    #     deployment_failed, endpoint_name = run_sagemaker_create_or_update_command(
    #         update_deployment_command
    #     )
    #     if not deployment_failed and endpoint_name:
    #         deployment_failed = test_deployment_result(endpoint_name, '1\n')
    # else:
    #     logger.info(
    #         'Deployment failed for updating env without changing BentoService'
    #     )

    if not deployment_failed:

        logger.info('UPDATED NEW BENTO FOR DEPLOYMENT')
        service_ver_two = UpdatedTestDeploymentService()
        saved_path = service_ver_two.save()

        loaded_ver_two_service = load(saved_path)
        bento_name = f'{loaded_ver_two_service.name}:{loaded_ver_two_service.version}'

        update_bento_version_deployment_command = [
            'bentoml',
            '--verbose',
            'deploy',
            'update',
            deployment_name,
            '-b',
            bento_name,
            '--wait',
        ]
        deployment_failed, endpoint_name = run_sagemaker_create_or_update_command(
            update_bento_version_deployment_command
        )
        if not deployment_failed and endpoint_name:
            deployment_failed = test_deployment_result(endpoint_name, '"dog"\n')
    else:
        logger.info('Deployment failed for updating BentoService')

    delete_deployment(deployment_name)

    logger.info('Finished')
    if deployment_failed:
        logger.info(
            'E2E update sagemaker deployment failed, fix the issues before releasing'
        )
    else:
        logger.info('E2E Sagemaker update deployment testing is successful')
