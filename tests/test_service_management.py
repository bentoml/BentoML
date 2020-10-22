# pylint: disable=redefined-outer-name
import os
import sys
import logging
import signal
import subprocess
import time
import uuid

import pytest

import docker

import bentoml
from bentoml.configuration import LAST_PYPI_RELEASE_VERSION
from bentoml.yatai.deployment.utils import ensure_docker_available_or_raise
from bentoml.service.management import push, pull, save, get_bento, list_bentos, load

logger = logging.getLogger('bentoml.test')


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


@pytest.fixture(scope='module')
def yatai_server_container():
    ensure_docker_available_or_raise()

    yatai_docker_image_tag = f'bentoml/yatai-service:{LAST_PYPI_RELEASE_VERSION}'
    container_name = f'e2e-test-yatai-service-container-{uuid.uuid4().hex[:6]}'
    port = '50055'
    command = [
        'docker',
        'run',
        '--rm',
        '--name',
        container_name,
        '-e',
        'BENTOML_HOME=/tmp',
        '-p',
        f'{port}:{port}',
        '-p',
        '3000:3000',
        yatai_docker_image_tag,
        '--grpc-port',
        port,
    ]

    logger.info(f"Starting docker container {container_name}: {command}")
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wait_until_container_ready(
        container_name, b'* Starting BentoML YataiService gRPC Server'
    )

    yield f'localhost:{port}'

    logger.info(f"Shutting down docker container: {container_name}")
    os.kill(docker_proc.pid, signal.SIGINT)


class TestModel(object):
    def predict(self, input_data):
        return int(input_data) * 2


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Requires docker, skipping test for Mac OS"
)
@pytest.mark.skipif('not psutil.POSIX')
def test_save_load(yatai_server_container, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=1)(
        example_bento_service_class
    )

    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)

    saved_path = save(svc, yatai_url=yatai_server_container)
    assert saved_path

    bento_service = load(f'{svc.name}:{svc.version}', yatai_url=yatai_server_container)
    assert bento_service.predict(1) == 2


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Requires docker, skipping test for Mac OS"
)
@pytest.mark.skipif('not psutil.POSIX')
def test_push(yatai_server_container, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=2)(
        example_bento_service_class
    )

    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = save(svc)

    pushed_path = push(f'{svc.name}:{svc.version}', yatai_url=yatai_server_container)
    assert pushed_path != saved_path


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Requires docker, skipping test for Mac OS"
)
@pytest.mark.skipif('not psutil.POSIX')
def test_pull(yatai_server_container, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=3)(
        example_bento_service_class
    )

    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = save(svc, yatai_url=yatai_server_container)

    pulled_local_path = pull(
        f'{svc.name}:{svc.version}', yatai_url=yatai_server_container
    )
    assert pulled_local_path != saved_path


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Requires docker, skipping test for Mac OS"
)
@pytest.mark.skipif('not psutil.POSIX')
def test_get(yatai_server_container, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=4)(
        example_bento_service_class
    )

    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    save(svc, yatai_url=yatai_server_container)
    svc_pb = get_bento(f'{svc.name}:{svc.version}', yatai_url=yatai_server_container)
    assert svc_pb.bento_service_metadata.name == svc.name
    assert svc_pb.bento_service_metadata.version == svc.version


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Requires docker, skipping test for Mac OS"
)
@pytest.mark.skipif('not psutil.POSIX')
def test_list(yatai_server_container, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=5)(
        example_bento_service_class
    )

    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    save(svc, yatai_url=yatai_server_container)

    bentos = list_bentos(bento_name=svc.name, yatai_url=yatai_server_container)
    assert len(bentos) == 5
