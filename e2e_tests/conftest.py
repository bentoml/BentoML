import logging
import os
import subprocess
import uuid
import time

import docker
import pytest
from sklearn import svm, datasets

from bentoml.deployment.aws_lambda import _cleanup_s3_bucket_if_exist
from bentoml.deployment.utils import ensure_docker_available_or_raise
from bentoml.utils.s3 import create_s3_bucket_if_not_exists
from bentoml.configuration import PREV_PYPI_RELEASE_VERSION
from bentoml.utils.tempdir import TempDirectory
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


def wait_for_docker_container_ready(container_name, check_message):
    docker_client = docker.from_env()

    start_time = time.time()
    while True:
        time.sleep(1)
        container_list = docker_client.containers.list(
            filters={'name': container_name, 'status': 'running'}
        )
        logger.info("Container list: " + str(container_list))
        if not container_list:
            # Raise timeout, if take more than 60 seconds
            if time.time() - start_time > 60:
                raise TimeoutError(f'Get container: {container_name} times out')
            else:
                continue

        assert len(container_list) == 1, 'should be only one container running'

        yatai_service_container = container_list[0]

        logger.info('container_log' + str(yatai_service_container.logs()))
        if check_message in yatai_service_container.logs():
            break


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
    wait_for_docker_container_ready(
        container_name, b'database system is ready to accept connections'
    )

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


@pytest.fixture(scope='session')
def temporary_yatai_service_url():
    ensure_docker_available_or_raise()
    docker_client = docker.from_env()
    local_bentoml_repo_path = os.path.abspath(os.path.join(__file__, '..', '..'))
    docker_tag = f'bentoml/yatai-service:local-bentoml-{uuid.uuid4().hex[:6]}'

    # Note: When set both `custom_context` and `fileobj`, docker api will not use the
    #       `path` provide... docker/api/build.py L138. The solution is create an actual
    #       Dockerfile along with path, instead of fileobj and custom_context.
    with TempDirectory() as temp_dir:
        temp_docker_file_path = os.path.join(temp_dir, 'Dockerfile')
        temp_docker_file = open(temp_docker_file_path, 'w')
        temp_docker_file.write(
            f'FROM bentoml/yatai-service:{PREV_PYPI_RELEASE_VERSION}\n'
        )
        temp_docker_file.write('ADD . /bentoml-local-repo\n')
        temp_docker_file.write('RUN pip install /bentoml-local-repo\n')
        temp_docker_file.close()
        logger.info('building docker image')
        docker_client.images.build(
            path=local_bentoml_repo_path,
            dockerfile=temp_docker_file_path,
            tag=docker_tag,
        )
        logger.info('complete build docker image')

        container_name = f'e2e-test-yatai-service-container-{uuid.uuid4().hex[:6]}'
        yatai_service_url = 'localhost:50051'

        command = [
            'docker',
            'run',
            '--rm',
            '--name',
            container_name,
            '-p',
            '50051:50051',
            '-p',
            '3000:3000',
            docker_tag,
            '--repo-base-url',
            '/tmp',
        ]

        logger.info(f'Running docker command {" ".join(command)}')

        docker_proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        wait_for_docker_container_ready(
            container_name, b'* Starting BentoML YataiService gRPC Server'
        )

        yield yatai_service_url
        docker_proc.terminate()
