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
    from scripts.e2e_tests.aws_sagemaker.utils import (
        run_sagemaker_create_or_update_command,
        test_deployment_result,
    )
    from scripts.e2e_tests.cli_operations import delete_deployment, delete_bento

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
        'sagemaker',
        'deploy',
        deployment_name,
        '-b',
        bento_name,
        '--region',
        region,
        '--api-name',
        'predict',
        '--num-of-gunicorn-workers-per-instance',
        '2',
        '--wait',
        '--verbose',
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

    delete_deployment('sagemaker', deployment_name)
    delete_bento(bento_name)

    logger.info('Finished')
    if deployment_failed:
        logger.info('E2E deployment failed, fix the issues before releasing')
    else:
        logger.info('E2E Sagemaker deployment testing is successful')
