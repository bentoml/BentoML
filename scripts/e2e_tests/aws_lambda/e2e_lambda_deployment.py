#!/usr/bin/env python

import subprocess
import logging
import uuid
import sys

import requests
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

    args = sys.argv
    bento_name = None
    if len(args) > 1:
        bento_name = args[1]
    if bento_name is None:
        logger.info('Training iris classifier with sklearn..')
        clf = svm.SVC(gamma='scale')
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        clf.fit(X, y)

        logger.info('Creating iris classifier BentoService bundle..')
        iris_clf_service = IrisClassifier()
        iris_clf_service.pack('clf', clf)
        saved_path = iris_clf_service.save()

        loaded_service = load(saved_path)
        sample_data = X[0:1]

        logger.info(
            'Result from sample data is: %s', str(loaded_service.predict(sample_data))
        )
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
        'aws-lambda',
        '--region',
        'us-west-2',
    ]
    logger.info(
        f"Running bentoml deploy command: {' '.join(create_deployment_command)}"
    )
    with subprocess.Popen(
        create_deployment_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        create_deployment_stdout = proc.stdout.read().decode('utf-8')
    logger.info('Finish deploying to AWS Lambda')
    logger.info(create_deployment_stdout)
    if create_deployment_stdout.startswith('Failed to create deployment'):
        deployment_failed = True
    create_deployment_output_list = create_deployment_stdout.split('\n')
    deployment_endpoint = ''
    for index, message in enumerate(create_deployment_output_list):
        if '"endpoints": [' in message:
            deployment_endpoint = (
                create_deployment_output_list[index + 1].strip().replace('"', '')
            )

    if not deployment_failed:
        logger.info('Test deployment with sample request')
        try:
            request_result = requests.post(
                deployment_endpoint,
                data=json.dumps(sample_data.tolist()),
                headers={'Content-Type': 'application/json'},
            )
            if request_result.status_code != 200:
                deployment_failed = True
            if request_result.content.decode('utf-8') != '[0]':
                logger.info(
                    'Test request failed. {}:{}'.format(
                        request_result.status_code,
                        request_result.content.decode('utf-8'),
                    )
                )
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
        logger.info('E2E Lambda deployment testing is successful')
