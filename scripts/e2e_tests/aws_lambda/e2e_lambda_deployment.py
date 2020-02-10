#!/usr/bin/env python

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
    from scripts.e2e_tests.aws_lambda.utils import (
        test_deployment_with_sample_data,
        run_lambda_create_or_update_command,
    )
    from scripts.e2e_tests.cli_operations import delete_deployment, delete_bento

    deployment_failed = False
    random_hash = uuid.uuid4().hex[:6]
    deployment_name = f'tests-lambda-e2e-{random_hash}'

    args = sys.argv
    bento_name = None
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
            deployment_endpoint, '[0]', json.dumps(sample_data.tolist())
        )

    delete_deployment('lambda', deployment_name)
    delete_bento(bento_name)

    logger.info('Finished')
    if deployment_failed:
        logger.info('E2E deployment failed, fix the issues before releasing')
    else:
        logger.info('E2E Lambda deployment testing is successful')
