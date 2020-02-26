#!/usr/bin/env python

import logging
import uuid
import sys


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


if __name__ == '__main__':
    from e2e_tests.aws_lambda.utils import (
        test_deployment_with_sample_data,
        run_lambda_create_or_update_command,
    )
    from e2e_tests.cli_operations import delete_deployment, delete_bento

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

    delete_deployment('lambda', deployment_name)
    delete_bento(bento_name)
    if updated_bento_name:
        delete_bento(updated_bento_name)

    logger.info('Finished')
    if deployment_failed:
        logger.info('E2E deployment failed, fix the issues before releasing')
    else:
        logger.info('E2E Lambda deployment testing is successful')
