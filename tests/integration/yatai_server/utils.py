import contextlib
import logging
import os
import subprocess
import uuid

import docker

from bentoml.configuration import LAST_PYPI_RELEASE_VERSION
from bentoml.utils import reserve_free_port
from bentoml.utils.tempdir import TempDirectory
from tests.integration.utils import wait_until_container_ready

logger = logging.getLogger("bentoml.test")


def build_yatai_service_image():
    docker_client = docker.from_env()
    local_bentoml_repo_path = os.path.abspath(__file__ + "/../../../../")
    yatai_docker_image_tag = f"bentoml/yatai-service:test-{uuid.uuid4().hex[:6]}"

    # Note: When set both `custom_context` and `fileobj`, docker api will not use the
    #   `path` provide... docker/api/build.py L138. The solution is create an actual
    #   Dockerfile along with path, instead of fileobj and custom_context.
    with TempDirectory() as temp_dir:
        temp_docker_file_path = os.path.join(temp_dir, "Dockerfile")
        with open(temp_docker_file_path, "w") as f:
            f.write(
                f"""\
FROM bentoml/yatai-service:{LAST_PYPI_RELEASE_VERSION}
ADD . /bentoml-local-repo
RUN pip install -U /bentoml-local-repo"""
            )
        logger.info(f"Building docker image {yatai_docker_image_tag}")
        docker_client.images.build(
            path=local_bentoml_repo_path,
            dockerfile=temp_docker_file_path,
            tag=yatai_docker_image_tag,
        )

    return yatai_docker_image_tag


# Cache the yatai docker image built for each test run session, since the source code
# of yatai will not be modified during a test run
_yatai_docker_image_tag = None


@contextlib.contextmanager
def yatai_service_container(db_url=None, repo_base_url=None):
    global _yatai_docker_image_tag  # pylint: disable=global-statement
    if _yatai_docker_image_tag is None:
        _yatai_docker_image_tag = build_yatai_service_image()

    docker_client = docker.from_env()
    container_name = f"yatai-test-{uuid.uuid4().hex[:6]}"
    yatai_server_command = ["bentoml", "yatai-service-start", "--no-ui"]
    if db_url:
        yatai_server_command.extend(["--db-url", db_url])
    if repo_base_url:
        yatai_server_command.extend(["--repo-base-url", repo_base_url])

    host = "127.0.0.1"
    with reserve_free_port(host) as free_port:
        # find free port on host
        port = free_port

    container = docker_client.containers.run(
        image=_yatai_docker_image_tag,
        remove=True,
        environment=["BENTOML_HOME=/tmp"],
        ports={"50051/tcp": (host, port)},
        command=yatai_server_command,
        name=container_name,
        detach=True,
    )

    wait_until_container_ready(
        container_name, "Starting BentoML YataiService gRPC Server"
    )
    yield f"{host}:{port}"

    logger.info(f"Shutting down docker container: {container_name}")
    container.kill()


@contextlib.contextmanager
def local_yatai_service_from_cli(db_url=None, repo_base_url=None, port=50051):
    yatai_server_command = [
        "bentoml",
        "yatai-service-start",
        "--no-ui",
        "--grpc-port",
        str(port),
    ]
    if db_url:
        yatai_server_command.extend(["--db-url", db_url])
    if repo_base_url:
        yatai_server_command.extend(["--repo-base-url", repo_base_url])
    logger.info(f'Starting local YataiServer {" ".join(yatai_server_command)}')
    proc = subprocess.Popen(
        yatai_server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    yatai_service_url = f"localhost:{port}"
    logger.info(f"Setting config yatai_service.url to: {yatai_service_url}")
    yield yatai_service_url
    proc.kill()
