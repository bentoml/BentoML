import logging
import subprocess
import uuid
import time

import docker
import pytest
from sklearn import svm, datasets

from bentoml.deployment.aws_lambda import _cleanup_s3_bucket_if_exist
from bentoml.deployment.utils import ensure_docker_available_or_raise
from bentoml.utils.s3 import create_s3_bucket_if_not_exists
from e2e_tests.iris_classifier_example import IrisClassifier
from e2e_tests.cli_operations import delete_bento
from e2e_tests.basic_bento_service_examples import (
    BasicBentoService,
    UpdatedBasicBentoService,
)

logger = logging.getLogger('bentoml.test')


@pytest.fixture()
def iris_clf_service():
    logger.debug('Training iris classifier with sklearn..')
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    logger.debug('Creating iris classifier BentoService bundle..')
    iris_clf_service = IrisClassifier()
    iris_clf_service.pack('clf', clf)
    iris_clf_service.save()

    bento_name = f'{iris_clf_service.name}:{iris_clf_service.version}'
    yield bento_name
    delete_bento(bento_name)


@pytest.fixture()
def basic_bentoservice_v1():
    logger.debug('Creating iris classifier BentoService bundle..')
    bento_svc = BasicBentoService()
    bento_svc.save()

    bento_name = f'{bento_svc.name}:{bento_svc.version}'
    yield bento_name
    delete_bento(bento_name)


@pytest.fixture()
def basic_bentoservice_v2():
    logger.debug('Creating iris classifier BentoService bundle..')
    bento_svc = UpdatedBasicBentoService()
    bento_svc.save()

    bento_name = f'{bento_svc.name}:{bento_svc.version}'
    yield bento_name
    delete_bento(bento_name)


@pytest.fixture(scope='session')
def temporary_docker_postgres_url():
    ensure_docker_available_or_raise()
    container_name = f'e2e-test-yatai-service-postgres-db-{uuid.uuid4().hex[:6]}'
    db_url = 'postgresql://postgres:postgres@localhost:5432/bentoml'

    command = [
        'docker',
        'run',
        '--rm',
        '--name',
        container_name,
        '-e',
        'POSTGRES_PASSWORD=postgres',
        '-p',
        '5432:5432',
        'postgres',
    ]
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # wait until postgres db is ready
    docker_client = docker.from_env()
    while True:
        time.sleep(1)
        container_list = docker_client.containers.list(
            filters={'name': container_name, 'status': 'running'}
        )
        logger.info("container_list: " + str(container_list))
        if not container_list:
            continue
        assert len(container_list) == 1, "should be only one container with name"

        postgres_container = container_list[0]
        logger.info("container_log:" + str(postgres_container.logs()))
        if (
            b'database system is ready to accept connections'
            in postgres_container.logs()
        ):
            break

    from sqlalchemy_utils import create_database

    create_database(db_url)
    yield db_url
    docker_proc.terminate()


@pytest.fixture()
def temporary_s3_bucket():
    random_hash = uuid.uuid4().hex[:6]
    bucket_name = f'e2e-yatai-server-{random_hash}'
    create_s3_bucket_if_not_exists(bucket_name, 'us-west-2')
    yield f's3://{bucket_name}/repo'
    _cleanup_s3_bucket_if_exist(bucket_name, 'us-west-2')
