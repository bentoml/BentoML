import logging
import os
import subprocess
import uuid

import docker
import pytest
from boto3 import Session
from sklearn import svm, datasets

from bentoml.deployment.utils import ensure_docker_available_or_raise
from bentoml.configuration import PREV_PYPI_RELEASE_VERSION
from bentoml.utils.s3 import create_s3_bucket_if_not_exists
from bentoml.utils.tempdir import TempDirectory
from e2e_tests.iris_classifier_example import IrisClassifier
from e2e_tests.cli_operations import delete_bento
from e2e_tests.basic_bento_service_examples import (
    BasicBentoService,
    UpdatedBasicBentoService,
)
from e2e_tests.utils import (
    cleanup_docker_containers,
    wait_for_docker_container_ready,
    modified_environ,
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
    iris_clf_service_ = IrisClassifier()
    iris_clf_service_.pack('clf', clf)
    iris_clf_service_.save()

    bento_name = f'{iris_clf_service_.name}:{iris_clf_service_.version}'
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


@pytest.fixture(scope="session")
def yatai_service_docker_image_tag():
    ensure_docker_available_or_raise()
    docker_client = docker.from_env()
    local_bentoml_repo_path = os.path.abspath(os.path.join(__file__, '..', '..'))
    docker_tag = f'bentoml/yatai-service:e2e-test-{uuid.uuid4().hex[:6]}'

    # Note: When set both `custom_context` and `fileobj`, docker api will not use the
    #       `path` provide... docker/api/build.py L138. The solution is create an actual
    #       Dockerfile along with path, instead of fileobj and custom_context.
    with TempDirectory() as temp_dir:
        temp_docker_file_path = os.path.join(temp_dir, 'Dockerfile')
        with open(temp_docker_file_path, 'w') as f:
            f.write(
                f"""\
    FROM bentoml/yatai-service:{PREV_PYPI_RELEASE_VERSION}
    ADD . /bentoml-local-repo
    RUN pip install /bentoml-local-repo
                """
            )
        docker_client.images.build(
            path=local_bentoml_repo_path,
            dockerfile=temp_docker_file_path,
            tag=docker_tag,
        )
        logger.info(f'complete build docker image {docker_tag}')
    return docker_tag


@pytest.fixture(scope='session')
def minio_container_service():
    ensure_docker_available_or_raise()

    container_name = f'bentoml-e2e-test-minio-container-{uuid.uuid4().hex[:6]}'
    minio_server_uri = 'http://127.0.0.1:9000'
    cleanup_docker_containers()
    command = [
        'docker',
        'run',
        '--rm',
        '--name',
        container_name,
        '-p',
        '9000:9000',
        'minio/minio',
        'server',
        '/data',
    ]
    logger.info(f'Running docker command {" ".join(command)}')

    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wait_for_docker_container_ready(
        container_name, b'Endpoint:  http://172.17.0.2:9000  http://127.0.0.1:9000'
    )
    bucket_name = 'yatai-e2e-test'
    with modified_environ(
        AWS_ACCESS_KEY_ID='minioadmin',
        AWS_SECRET_ACCESS_KEY='minioadmin',
        AWS_REGION='us-east-1',
        BENTOML__YATAI_SERVICE__S3_ENDPOINT_URL=minio_server_uri,
    ):
        create_s3_bucket_if_not_exists(bucket_name, 'us-east-1')

    yield {
        'url': minio_server_uri,
        'bucket_name': bucket_name,
        'container_name': container_name,
    }
    docker_proc.terminate()


@pytest.fixture(scope='session')
def postgres_docker_container():
    ensure_docker_available_or_raise()
    cleanup_docker_containers()
    container_name = (
        f'bentoml-e2e-test-yatai-service-postgres-db-{uuid.uuid4().hex[:6]}'
    )

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
    logger.debug(f'Docker command: {" ".join(command)}')
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wait_for_docker_container_ready(
        container_name, b'database system is ready to accept connections'
    )

    from sqlalchemy_utils import create_database

    # Note: The host of postgres in create_database is different from the db_url that
    # an YataiService container will use. Using `--link` option on docker run to map
    # the network ports on postgres to yatai service container
    create_database(f'postgresql://postgres:postgres@localhost:5432/bentoml')
    db_url = 'postgresql://postgres:postgres@postgres-container:5432/bentoml'
    yield {'url': db_url, 'container_name': container_name}
    docker_proc.terminate()
