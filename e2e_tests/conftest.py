import logging
import subprocess
import uuid
from time import sleep

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


def wait_until_container_is_running(container_name):
    docker_client = docker.from_env()
    is_docker_running = False
    while not is_docker_running:
        logger.info('Fetching running container list')
        container_list = docker_client.containers.list(filters={'status': 'running'})
        for container in container_list:
            if container.name == container_name:
                is_docker_running = True
                break
        sleep(5)
    return


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
def temporary_docker_postgres_url(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp('yatai_db')
    ensure_docker_available_or_raise()
    container_name = 'e2e-yatai-pg-docker'

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
        '-v',
        f'{tmpdir}:/var/lib/postgresql/data',
        'postgres',
    ]
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wait_until_container_is_running(container_name)
    yield 'postgresql://postgres:postgres@localhost:5432/bentoml'
    docker_proc.terminate()


@pytest.fixture()
def temporary_s3_bucket():
    random_hash = uuid.uuid4().hex[:6]
    bucket_name = f'e2e-yatai-server-{random_hash}'
    create_s3_bucket_if_not_exists(bucket_name, 'us-west-2')
    yield f's3://{bucket_name}/repo'
    _cleanup_s3_bucket_if_exist(bucket_name, 'us-west-2')
