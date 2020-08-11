# pylint: disable=redefined-outer-name
import time
import urllib
import logging
from contextlib import contextmanager

import pytest
import bentoml

from .example_service import gen_test_bundle


logger = logging.getLogger("bentoml.tests")


@pytest.fixture(params=[True, False], scope="session")
def enable_microbatch(request):
    pytest.enable_microbatch = request.param
    return request.param


@pytest.fixture(autouse=True, scope='session')
def image(tmpdir_factory):
    bundle_dir = tmpdir_factory.mktemp('test_bundle')
    bundle_path = str(bundle_dir)
    gen_test_bundle(bundle_path)

    with build_api_server_docker_image(bundle_path, "example_service") as image:
        yield image


@pytest.fixture(autouse=True)
def host(image, enable_microbatch):
    with run_api_server_docker_container(image, enable_microbatch) as host:
        yield host


def _wait_until_api_server_ready(host_url, timeout, container, check_interval=1):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if (
                urllib.request.urlopen(f'http://{host_url}/healthz', timeout=1).status
                == 200
            ):
                break
            else:
                logger.info("Waiting for host %s to be ready..", host_url)
                time.sleep(check_interval)
        except Exception as e:  # pylint:disable=broad-except
            logger.info(f"Error caught waiting for host {host_url} to be ready: '{e}'")
            time.sleep(check_interval)
        finally:
            container_logs = container.logs()
            if container_logs:
                logger.info(f"Container {container.id} logs:")
                for log_record in container_logs.decode().split('\r\n'):
                    logger.info(f">>> {log_record}")
    else:
        raise AssertionError(
            f"Timed out waiting {timeout} seconds for Server {host_url} to be ready"
        )


@contextmanager
def run_api_server_docker_container(image, enable_microbatch=False, timeout=60):
    """
    yields the host URL
    """
    import docker

    client = docker.from_env()

    with bentoml.utils.reserve_free_port() as port:
        pass
    if enable_microbatch:
        command_args = "--enable-microbatch --workers 1"
    else:
        command_args = "--workers 1"
    try:
        container = client.containers.run(
            image=image.id,
            command=command_args,
            auto_remove=True,
            tty=True,
            ports={'5000/tcp': port},
            detach=True,
        )
        host_url = f"127.0.0.1:{port}"
        _wait_until_api_server_ready(host_url, timeout, container)
        yield host_url
    finally:
        container.stop()
        time.sleep(1)  # make sure container stopped & deleted


@contextmanager
def build_api_server_docker_image(saved_bundle_path, image_tag):
    import docker

    client = docker.from_env()
    try:
        logger.info(
            f"Building API server docker image from build context: {saved_bundle_path}"
        )
        image = client.images.build(path=saved_bundle_path, tag=image_tag, rm=True)[0]
        yield image
    finally:
        client.images.remove(image.id)
