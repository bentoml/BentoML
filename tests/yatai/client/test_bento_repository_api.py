# pylint: disable=redefined-outer-name
import logging
import os
import signal
import subprocess
import sys
import time
import uuid

import psutil  # noqa # pylint: disable=unused-import
import pytest

import bentoml
import docker
from bentoml.configuration import LAST_PYPI_RELEASE_VERSION
from bentoml.exceptions import InvalidArgument
from bentoml.saved_bundle.loader import load_from_dir
from bentoml.yatai.client import get_yatai_client
from bentoml.yatai.deployment.docker_utils import ensure_docker_available_or_raise
from bentoml.yatai.label_store import _validate_labels

logger = logging.getLogger('bentoml.test')


def test_validate_labels_fails():
    with pytest.raises(InvalidArgument):
        _validate_labels(
            {'this_is_a_super_long_key_name_it_will_be_more_than_the_max_allowed': 'v'}
        )
    with pytest.raises(InvalidArgument):
        _validate_labels({'key_contains!': 'value'})
    with pytest.raises(InvalidArgument):
        _validate_labels({'key': 'value-contains?'})
    with pytest.raises(InvalidArgument):
        _validate_labels({'key nop': 'value'})
    with pytest.raises(InvalidArgument):
        _validate_labels({'key': '1', 'key3@#!$': 'value'})
    with pytest.raises(InvalidArgument):
        _validate_labels({'key': 'cant_end_with_symbol_'})


def test_validate_labels_pass():
    _validate_labels({'long_key_title': 'some_value', 'another_key': "value"})
    _validate_labels({'long_key-title': 'some_value-inside.this'})
    _validate_labels({'create_by': 'admin', 'py.version': '3.6.8'})


def wait_until_container_ready(container_name, check_message, timeout_seconds=120):
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
        container_name, b'* Starting BentoML YataiService gRPC Server', 120
    )

    yield f'localhost:{port}'

    logger.info(f"Shutting down docker container: {container_name}")
    os.kill(docker_proc.pid, signal.SIGINT)


class TestModel(object):
    def predict(self, input_data):
        return int(input_data) * 2


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Requires docker, skipping test for Mac OS on Github Action",
)
@pytest.mark.skipif('not psutil.POSIX')
def test_save_load(yatai_server_container, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=1)(
        example_bento_service_class
    )

    yc = get_yatai_client(yatai_server_container)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)

    saved_path = svc.save(yatai_url=yatai_server_container)
    assert saved_path

    bento_pb = yc.repository.get(f'{svc.name}:{svc.version}')
    bento_service = load_from_dir(bento_pb.uri.uri)
    assert bento_service.predict(1) == 2


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Requires docker, skipping test for Mac OS on Github Action",
)
@pytest.mark.skipif('not psutil.POSIX')
def test_push(yatai_server_container, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=2)(
        example_bento_service_class
    )

    yc = get_yatai_client(yatai_server_container)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = svc.save()

    pushed_path = yc.repository.push(f'{svc.name}:{svc.version}')
    assert pushed_path != saved_path


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Requires docker, skipping test for Mac OS on Github Action",
)
@pytest.mark.skipif('not psutil.POSIX')
def test_pull(yatai_server_container, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=3)(
        example_bento_service_class
    )

    yc = get_yatai_client(yatai_server_container)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = svc.save(yatai_url=yatai_server_container)

    pulled_local_path = yc.repository.pull(f'{svc.name}:{svc.version}')
    assert pulled_local_path != saved_path


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Requires docker, skipping test for Mac OS for Github Action",
)
@pytest.mark.skipif('not psutil.POSIX')
def test_get(yatai_server_container, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=4)(
        example_bento_service_class
    )

    yc = get_yatai_client(yatai_server_container)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    svc.save(yatai_url=yatai_server_container)
    svc_pb = yc.repository.get(f'{svc.name}:{svc.version}')
    assert svc_pb.bento_service_metadata.name == svc.name
    assert svc_pb.bento_service_metadata.version == svc.version


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Requires docker, skipping test for Mac OS for Github Action",
)
@pytest.mark.skipif('not psutil.POSIX')
def test_list(yatai_server_container, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=5)(
        example_bento_service_class
    )
    yc = get_yatai_client(yatai_server_container)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    svc.save(yatai_url=yatai_server_container)

    bentos = yc.repository.list(bento_name=svc.name)
    assert len(bentos) == 5


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Requires docker, skipping test for Mac OS for Github Action",
)
@pytest.mark.skipif('not psutil.POSIX')
def test_load(yatai_server_container, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=5)(
        example_bento_service_class
    )
    yc = get_yatai_client(yatai_server_container)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    svc.save(yatai_url=yatai_server_container)

    loaded_svc = yc.repository.load(f'{svc.name}:{svc.version}')
    assert loaded_svc.name == svc.name


@pytest.mark.skipif('not psutil.POSIX')
def test_load_from_dir(example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=6)(
        example_bento_service_class
    )
    yc = get_yatai_client()
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = svc.save()

    loaded_svc = yc.repository.load(saved_path)
    assert loaded_svc.name == svc.name
