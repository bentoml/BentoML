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
        '--region',
        region,
        '--api-name',
        'predict',
        '--gunicorn-workers-per-instance',
        '2',
    ]
    logger.info(
        f"Running bentoml deploy command: {' '.join(create_deployment_command)}"
    )
    with subprocess.Popen(
        create_deployment_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        create_deployment_stdout = proc.stdout.read().decode('utf-8')
    logger.info('Finish deploying to AWS Sagemaker')
    logger.info(create_deployment_stdout)
    if create_deployment_stdout.startswith('Failed to create deployment'):
        deployment_failed = True
    create_deployment_output_list = create_deployment_stdout.split('\n')
    deployment_endpoint = ''
    for index, message in enumerate(create_deployment_output_list):
        if '"EndpointName":' in message:
            endpoint_name = message.split(':')[1].strip(',').replace('"', '')
            deployment_endpoint = (
                f'https://runtime.sagemaker.{region}.amazonaws.com/'
                f'endpoints/{endpoint_name}/invocations'
            )

    if not deployment_failed and endpoint_name:
        logger.info(f'Test deployment with sample request for {endpoint_name}')
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
                f'"{json.dumps(sample_data.tolist())}"',
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
            logger.info('Prediction Result: %s', result.stdout.decode('utf-8'))
            if '[\n  0\n]\n' == result.stdout.decode('utf-8'):
                deployment_failed = False
            else:
                deployment_failed = True
        except Exception as e:
            logger.error(str(e))
            deployment_failed = True

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
        logger.info('E2E deployment failed, fix the issues before releasing')
    else:
        logger.info('E2E Sagemaker deployment testing is successful')
