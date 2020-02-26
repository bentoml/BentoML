#!/usr/bin/env python

import logging
import uuid

from bentoml import BentoService, load, api
from bentoml.handlers import DataframeHandler


logger = logging.getLogger('bentoml.test')


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
    from e2e_tests.aws_sagemaker.utils import (
        run_sagemaker_create_or_update_command,
        test_deployment_result,
    )
    from e2e_tests.cli_operations import delete_deployment, delete_bento

    deployment_failed = False
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-sagemaker-update-e2e-{random_hash}'
    region = 'us-west-2'

    logger.info('Creating version one BentoService bundle..')
    service_ver_one = TestDeploymentService()
    saved_path = service_ver_one.save()

    loaded_ver_one_service = load(saved_path)
    bento_name = f'{loaded_ver_one_service.name}:{loaded_ver_one_service.version}'
    create_deployment_command = [
        'bentoml',
        'sagemaker',
        'deploy',
        deployment_name,
        '-b',
        bento_name,
        '--api-name',
        'classify',
        '--verbose',
    ]
    deployment_failed, endpoint_name = run_sagemaker_create_or_update_command(
        create_deployment_command
    )

    if not deployment_failed and endpoint_name:
        deployment_failed = test_deployment_result(endpoint_name, '"cat"\n')
    else:
        deployment_failed = True
        logger.info('Deployment failed for creating deployment')

    if not deployment_failed:
        logger.info('UPDATED NEW BENTO FOR DEPLOYMENT')
        service_ver_two = UpdatedTestDeploymentService()
        saved_path = service_ver_two.save()

        loaded_ver_two_service = load(saved_path)
        updated_bento_name = (
            f'{loaded_ver_two_service.name}:{loaded_ver_two_service.version}'
        )

        update_bento_version_deployment_command = [
            'bentoml',
            'sagemaker',
            'update',
            deployment_name,
            '-b',
            updated_bento_name,
            '--wait',
            '--verbose',
        ]
        deployment_failed, endpoint_name = run_sagemaker_create_or_update_command(
            update_bento_version_deployment_command
        )
        if not deployment_failed and endpoint_name:
            deployment_failed = test_deployment_result(endpoint_name, '"dog"\n')
    else:
        logger.info('Deployment failed for updating BentoService')

    delete_deployment('sagemaker', deployment_name)
    delete_bento(bento_name)

    if updated_bento_name:
        delete_bento(updated_bento_name)

    logger.info('Finished')
    if deployment_failed:
        logger.info(
            'E2E update sagemaker deployment failed, fix the issues before releasing'
        )
    else:
        logger.info('E2E Sagemaker update deployment testing is successful')
