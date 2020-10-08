import logging
import subprocess
import uuid
import time
import os
import signal
import sys

import docker
import pytest
from sklearn import svm, datasets


# Append local bentoml repository path which contains the 'e2e_tests/' direcotry
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from bentoml.yatai.deployment.utils import ensure_docker_available_or_raise

from e2e_tests.iris_classifier_example import IrisClassifier
from e2e_tests.cli_operations import delete_bento
from e2e_tests.sample_bento_service import (
    SampleBentoService,
    UpdatedSampleBentoService,
)


logger = logging.getLogger('bentoml.test')


@pytest.fixture()
def iris_clf_service():
    logger.debug('Training iris classifier with sklearn..')
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    logger.debug('Creating iris classifier service saved bundle..')
    iris_clf_service_ = IrisClassifier()
    iris_clf_service_.pack('clf', clf)
    iris_clf_service_.save()

    bento_name = f'{iris_clf_service_.name}:{iris_clf_service_.version}'
    yield bento_name
    delete_bento(bento_name)


@pytest.fixture()
def basic_bentoservice_v1():
    logger.debug('Creating basic_bentoservice_v1 saved bundle..')
    bento_svc = SampleBentoService()
    bento_svc.save()

    bento_name = f'{bento_svc.name}:{bento_svc.version}'
    yield bento_name
    delete_bento(bento_name)


@pytest.fixture()
def basic_bentoservice_v2():
    logger.debug('Creating basic_bentoservice_v2 saved bundle..')
    bento_svc = UpdatedSampleBentoService()
    bento_svc.save()

    bento_name = f'{bento_svc.name}:{bento_svc.version}'
    yield bento_name
    delete_bento(bento_name)


def wait_until_container_ready(container_name, check_message, timeout_seconds=60):
    docker_client = docker.from_env()

    start_time = time.time()
    while True:
        time.sleep(1)
        container_list = docker_client.containers.list(filters={'name': container_name})
        logger.info("Container list: " + str(container_list))
        if not container_list:
            # Raise timeout, if exceeds timeout limit
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f'Get container: {container_name} timed out')
            else:
                continue

        assert (
            len(container_list) == 1
        ), f'should be exact one container with name {container_name}'

        if check_message in container_list[0].logs():
            break


@pytest.fixture(scope='session')
def postgres_db_container_url():
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
    logger.info(f"Starting Postgres Server container {container_name}: {command}")
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wait_until_container_ready(
        container_name, b'database system is ready to accept connections'
    )

    from sqlalchemy_utils import create_database

    create_database(db_url)
    yield db_url

    logger.info(f"Shutting down Postgres Server container: {container_name}")
    os.kill(docker_proc.pid, signal.SIGINT)
