import contextlib
import logging
import os
import signal
import subprocess
import time
import uuid

import docker

from bentoml.configuration import LAST_PYPI_RELEASE_VERSION
from bentoml.utils.tempdir import TempDirectory
from bentoml.yatai.deployment.utils import ensure_docker_available_or_raise

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


@contextlib.contextmanager
def local_yatai_server():
    ensure_docker_available_or_raise()
    docker_client = docker.from_env()
    local_bentoml_repo_path = os.path.abspath(__file__ + "/../../../../")
    yatai_docker_image_tag = f'bentoml/yatai-service:e2e-test-{uuid.uuid4().hex[:6]}'

    # Note: When set both `custom_context` and `fileobj`, docker api will not use the
    #       `path` provide... docker/api/build.py L138. The solution is create an actual
    #       Dockerfile along with path, instead of fileobj and custom_context.
    with TempDirectory() as temp_dir:
        temp_docker_file_path = os.path.join(temp_dir, 'Dockerfile')
        with open(temp_docker_file_path, 'w') as f:
            f.write(
                f"""\
FROM bentoml/yatai-service:{LAST_PYPI_RELEASE_VERSION}
ADD . /bentoml-local-repo
RUN pip install /bentoml-local-repo
            """
            )
        logger.info(f'building docker image {yatai_docker_image_tag}')
        docker_client.images.build(
            path=local_bentoml_repo_path,
            dockerfile=temp_docker_file_path,
            tag=yatai_docker_image_tag,
        )

        container_name = f'e2e-test-yatai-service-container-{uuid.uuid4().hex[:6]}'
        yatai_service_url = 'localhost:50051'
        command = [
            'docker',
            'run',
            '--rm',
            '--name',
            container_name,
            '-e',
            'BENTOML_HOME=/tmp',
            '-p',
            '50051:50051',
            '-p',
            '3000:3000',
            yatai_docker_image_tag,
        ]

        logger.info(f"Starting docker container {container_name}: {command}")
        docker_proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        wait_until_container_ready(
            container_name, b'* Starting BentoML YataiService gRPC Server'
        )

        yield yatai_service_url

        logger.info(f"Shutting down docker container: {container_name}")
        os.kill(docker_proc.pid, signal.SIGINT)
