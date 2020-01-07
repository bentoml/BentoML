#!/usr/bin/env python

import subprocess
import logging
import uuid
import sys

import json
from sklearn import svm, datasets

from bentoml import BentoService, load, api, env, artifacts
from bentoml.artifact import PickleArtifact
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


@artifacts([PickleArtifact('clf')])
@env(pip_dependencies=['scikit-learn'])
class IrisClassifier(BentoService):
    @api(DataframeHandler)
    def predict(self, df):
        return self.artifacts.clf.predict(df)


if __name__ == '__main__':
    deployment_failed = False
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-lambda-e2e-{random_hash}'
    region = 'us-west-2'

    args = sys.argv
    bento_name = None
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    sample_data = X[0:1]
    if len(args) > 1:
        bento_name = args[1]
    if bento_name is None:
        logger.info('Training iris classifier with sklearn..')
        clf = svm.SVC(gamma='scale')
        clf.fit(X, y)

        logger.info('Creating iris classifier BentoService bundle..')
        iris_clf_service = IrisClassifier()
        iris_clf_service.pack('clf', clf)
        saved_path = iris_clf_service.save()

        loaded_service = load(saved_path)
        bento_name = f'{loaded_service.name}:{loaded_service.version}'
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
        # '--region',
        # region,
        '--api-name',
        'predict',
        '--num-of-gunicorn-workers-per-instance',
        '2',
        '--wait',
    ]
    deployment_failed, endpoint_name = run_sagemaker_create_or_update_command(
        create_deployment_command
    )
    logger.info(f'Finished create deployment {deployment_name}')
    logger.info(
        f'RESULT FROM CREATE DEPLOYMENT: {deployment_failed}. '
        f'Endpoint is {endpoint_name}'
    )

    if not deployment_failed and endpoint_name:
        deployment_failed = test_deployment_result(
            endpoint_name, '[\n  0\n]\n', f'"{json.dumps(sample_data.tolist())}"'
        )
    else:
        logger.info('Create deployment failed')
        deployment_failed = True

    delete_deployment(deployment_name)

    logger.info('Finished')
    if deployment_failed:
        logger.info('E2E deployment failed, fix the issues before releasing')
    else:
        logger.info('E2E Sagemaker deployment testing is successful')
