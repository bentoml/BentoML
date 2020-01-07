#!/usr/bin/env python

import subprocess
import logging
import uuid

from bentoml import BentoService, load, api
from bentoml.handlers import DataframeHandler
from scripts.e2e_tests.aws_sagemaker.utils import (
    run_sagemaker_create_or_update_command,
    test_deployment_result,
)

logger = logging.getLogger('bentoml.test')


class TestUpdateDeploymentService(BentoService):
    @api(DataframeHandler)
    def predict(self, df):
        return 1

    @api(DataframeHandler)
    def classify(self, df):
        return 'cat'


if __name__ == '__main__':
    deployment_failed = False
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-sagemaker-update-e2e-{random_hash}'
    region = 'us-west-2'

    logger.info('Creating version one BentoService bundle..')
    service_ver_one = TestUpdateDeploymentService()
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
        'predict',
    ]
    deployment_failed, endpoint_name = run_sagemaker_create_or_update_command(
        create_deployment_command
    )

    if not deployment_failed and endpoint_name:
        deployment_failed = test_deployment_result(endpoint_name, '1')

    update_deployment_command = [
        'bentoml',
        '--verbose',
        'deploy',
        'update',
        deployment_name,
        '--api-name',
        'classify',
    ]
    deployment_failed, endpoint_name = run_sagemaker_create_or_update_command(
        update_deployment_command
    )
    if not deployment_failed and endpoint_name:
        deployment_failed = test_deployment_result(endpoint_name, 'cat')

    class TestUpdateDeploymentService(BentoService):
        @api(DataframeHandler)
        def predict(self, df):
            return 1

        @api(DataframeHandler)
        def classify(self, df):
            # change result to dog
            return 'dog'

    logger.info('Creating version two BentoService bundle..')
    service_ver_two = TestUpdateDeploymentService()
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
    ]
    deployment_failed, endpoint_name = run_sagemaker_create_or_update_command(
        update_bento_version_deployment_command
    )
    if not deployment_failed and endpoint_name:
        deployment_failed = test_deployment_result(endpoint_name, 'dog')

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

    logger.info('Finished')
    if deployment_failed:
        logger.info(
            'E2E update sagemaker deployment failed, fix the issues before releasing'
        )
    else:
        logger.info('E2E Sagemaker update deployment testing is successful')
