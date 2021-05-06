# pylint: disable=redefined-outer-name
import logging
import os
import sys
import uuid

import docker
import psutil
import pytest

import bentoml
from bentoml.configuration import LAST_PYPI_RELEASE_VERSION
from bentoml.saved_bundle.loader import load_from_dir
from bentoml.utils.tempdir import TempDirectory
from bentoml.yatai.client import get_yatai_client
from bentoml.yatai.deployment.docker_utils import ensure_docker_available_or_raise
from tests.yatai.local_yatai_service import wait_until_container_ready

logger = logging.getLogger('bentoml.test')

if sys.platform == "darwin" or not psutil.POSIX:
    pytest.skip(
        "Requires docker, skipping test for Mac OS on Github Actions",
        allow_module_level=True,
    )


@pytest.fixture(scope='session')
def local_yatai_service_url():
    ensure_docker_available_or_raise()
    docker_client = docker.from_env()
    local_bentoml_repo_path = os.path.abspath(__file__ + "/../../../../")
    yatai_docker_image_tag = f'bentoml/yatai-service:e2e-test-{uuid.uuid4().hex[:6]}'

    # Note: When set both `custom_context` and `fileobj`, docker api will not use the
    #   `path` provide... docker/api/build.py L138. The solution is create an actual
    #   Dockerfile along with path, instead of fileobj and custom_context.
    with TempDirectory() as temp_dir:
        temp_docker_file_path = os.path.join(temp_dir, 'Dockerfile')
        with open(temp_docker_file_path, 'w') as f:
            f.write(
                f"""\
FROM bentoml/yatai-service:{LAST_PYPI_RELEASE_VERSION}
ADD . /bentoml-local-repo
RUN pip install -U /bentoml-local-repo
            """
            )
        logger.info(f'building docker image {yatai_docker_image_tag}')
        docker_client.images.build(
            path=local_bentoml_repo_path,
            dockerfile=temp_docker_file_path,
            tag=yatai_docker_image_tag,
        )

        container_name = f'yatai-service-container-{uuid.uuid4().hex[:6]}'
        yatai_server_command = ['bentoml', 'yatai-service-start', '--no-ui']
        port = 50051
        yatai_service_url = f'localhost:{port}'
        container = docker_client.containers.run(
            image=yatai_docker_image_tag,
            environment=['BENTOML_HOME=/tmp'],
            ports={f'{port}/tcp': port},
            command=yatai_server_command,
            name=container_name,
            detach=True,
        )

        wait_until_container_ready(container)
        yield yatai_service_url

        logger.info(f"Shutting down docker container: {container_name}")
        container.kill()


class TestModel(object):
    def predict(self, input_data):
        return int(input_data) * 2


def test_save_load(local_yatai_service_url, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=1)(
        example_bento_service_class
    )

    yc = get_yatai_client(local_yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)

    saved_path = svc.save(yatai_url=local_yatai_service_url)
    assert saved_path

    bento_pb = yc.repository.get(f'{svc.name}:{svc.version}')
    with TempDirectory() as temp_dir:
        new_temp_dir = os.path.join(temp_dir, uuid.uuid4().hex[:12])
        yc.repository.download_to_directory(bento_pb, new_temp_dir)
        bento_service = load_from_dir(new_temp_dir)
        assert bento_service.predict(1) == 2


def test_push(local_yatai_service_url, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=2)(
        example_bento_service_class
    )

    yc = get_yatai_client(local_yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = svc.save()

    pushed_path = yc.repository.push(f'{svc.name}:{svc.version}')
    assert pushed_path != saved_path


def test_pull(local_yatai_service_url, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=3)(
        example_bento_service_class
    )

    yc = get_yatai_client(local_yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = svc.save(yatai_url=local_yatai_service_url)

    pulled_local_path = yc.repository.pull(f'{svc.name}:{svc.version}')
    assert pulled_local_path != saved_path


def test_push_with_labels(local_yatai_service_url, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=2)(
        example_bento_service_class
    )

    yc = get_yatai_client(local_yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = svc.save(labels={'foo': 'bar', 'abc': '123'})

    pushed_path = yc.repository.push(f'{svc.name}:{svc.version}')
    assert pushed_path != saved_path
    remote_bento_pb = yc.repository.get(f'{svc.name}:{svc.version}')
    assert remote_bento_pb.bento_service_metadata.labels
    labels = dict(remote_bento_pb.bento_service_metadata.labels)
    assert labels['foo'] == 'bar'
    assert labels['abc'] == '123'


def test_pull_with_labels(local_yatai_service_url, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=3)(
        example_bento_service_class
    )

    yc = get_yatai_client(local_yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    saved_path = svc.save(
        yatai_url=local_yatai_service_url, labels={'foo': 'bar', 'abc': '123'}
    )

    pulled_local_path = yc.repository.pull(f'{svc.name}:{svc.version}')
    assert pulled_local_path != saved_path
    local_yc = get_yatai_client()
    local_bento_pb = local_yc.repository.get(f'{svc.name}:{svc.version}')
    assert local_bento_pb.bento_service_metadata.labels
    labels = dict(local_bento_pb.bento_service_metadata.labels)
    assert labels['foo'] == 'bar'
    assert labels['abc'] == '123'


def test_get(local_yatai_service_url, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=4)(
        example_bento_service_class
    )

    yc = get_yatai_client(local_yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    svc.save(yatai_url=local_yatai_service_url)
    svc_pb = yc.repository.get(f'{svc.name}:{svc.version}')
    assert svc_pb.bento_service_metadata.name == svc.name
    assert svc_pb.bento_service_metadata.version == svc.version


def test_list(local_yatai_service_url, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=5)(
        example_bento_service_class
    )
    yc = get_yatai_client(local_yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    svc.save(yatai_url=local_yatai_service_url)

    bentos = yc.repository.list(bento_name=svc.name)
    assert len(bentos) == 7


def test_load(local_yatai_service_url, example_bento_service_class):
    example_bento_service_class = bentoml.ver(major=2, minor=5)(
        example_bento_service_class
    )
    yc = get_yatai_client(local_yatai_service_url)
    test_model = TestModel()
    svc = example_bento_service_class()
    svc.pack('model', test_model)
    svc.save(yatai_url=local_yatai_service_url)

    loaded_svc = yc.repository.load(f'{svc.name}:{svc.version}')
    assert loaded_svc.name == svc.name


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
