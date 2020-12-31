# pylint: disable=redefined-outer-name
import logging
import os
import subprocess
import sys
import threading
import time
import urllib
from contextlib import contextmanager

import bentoml
from bentoml.utils import cached_contextmanager

logger = logging.getLogger("bentoml.tests")


def _wait_until_api_server_ready(host_url, timeout, container=None, check_interval=1):
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
            if container:
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
def export_service_bundle(bento_service):
    """
    Export a bentoml service to a temporary directory, yield the path.
    Delete the temporary directory on close.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as path:
        bento_service.save_to_dir(path)
        yield path


@cached_contextmanager("{saved_bundle_path}, {image_tag}")
def build_api_server_docker_image(saved_bundle_path, image_tag="test_bentoml_server"):
    """
    Build the docker image for a saved bentoml bundle, yield the docker image object.
    """

    import docker

    client = docker.from_env()
    logger.info(
        f"Building API server docker image from build context: {saved_bundle_path}"
    )
    try:
        image, _ = client.images.build(path=saved_bundle_path, tag=image_tag, rm=False)
        yield image
        client.images.remove(image.id)
    except docker.errors.BuildError as e:
        for line in e.build_log:
            if 'stream' in line:
                print(line['stream'].strip())
        raise


@cached_contextmanager("{image.id}, {enable_microbatch}")
def run_api_server_docker_container(image, enable_microbatch=False, timeout=60):
    """
    Launch a bentoml service container from a docker image, yields the host URL.
    """
    import docker

    client = docker.from_env()

    with bentoml.utils.reserve_free_port() as port:
        pass
    if enable_microbatch:
        command_args = "--enable-microbatch --workers 1 --mb-max-batch-size 2048"
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
            remove=True,
        )
        host_url = f"127.0.0.1:{port}"
        _wait_until_api_server_ready(host_url, timeout, container)
        yield host_url
    finally:
        print(container.logs())
        container.stop()
        time.sleep(1)  # make sure container stopped & deleted


@contextmanager
def run_api_server(bundle_path, enable_microbatch=False, timeout=10, init_cmd=None):
    """
    Launch a bentoml service directly by the bentoml CLI, yields the host URL.
    """
    if init_cmd:
        with subprocess.Popen(init_cmd) as p:
            p.wait(60)

    with bentoml.utils.reserve_free_port() as port:
        my_env = os.environ.copy()
        cmd = [sys.executable, "-m", "bentoml", "serve-gunicorn"]
        if port:
            cmd += ['--port', f'{port}']
        if enable_microbatch:
            cmd += ['--enable-microbatch']
        cmd += [bundle_path, "--workers", "1"]

    def print_log(p):
        for line in p.stdout:
            print(line.decode(), end='')

    with subprocess.Popen(
        cmd,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        env=my_env,
    ) as p:
        host_url = f"127.0.0.1:{port}"
        threading.Thread(target=print_log, args=(p,), daemon=True).start()
        _wait_until_api_server_ready(host_url, timeout=timeout)
        yield host_url
        p.terminate()
