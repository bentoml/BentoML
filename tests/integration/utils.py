# pylint: disable=redefined-outer-name
from contextlib import contextmanager
import logging
import os
import subprocess
import sys
import threading
import time
import urllib

import bentoml
from bentoml.utils import cached_contextmanager

logger = logging.getLogger("bentoml.tests")


def _wait_until_api_server_ready(host_url, timeout, container=None, check_interval=1):
    start_time = time.time()
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    ex = None
    while time.time() - start_time < timeout:
        try:
            if opener.open(f'http://{host_url}/healthz', timeout=1).status == 200:
                return
            elif container and container.status != "running":
                break
            else:
                logger.info("Waiting for host %s to be ready..", host_url)
                time.sleep(check_interval)
        except Exception as e:  # pylint:disable=broad-except
            logger.info(f"retrying to connect to the host {host_url}...")
            ex = e
            time.sleep(check_interval)
        finally:
            if container:
                container_logs = container.logs()
                if container_logs:
                    logger.info(f"Container {container.id} logs:")
                    for log_record in container_logs.decode().split('\r\n'):
                        logger.info(f">>> {log_record}")
    else:
        logger.info("Timeout!")
        raise AssertionError(
            f"Timed out waiting {timeout} seconds for Server {host_url} to be ready, "
            f"exception: {ex}"
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
def run_api_server_docker_container(
    image, enable_microbatch=False, config_file=None, timeout=60
):
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

    if config_file is not None:
        command_args += " -e BENTOML_CONFIG=/etc/bentoml_config.yml"
        volumes = {
            os.path.abspath(config_file): {
                "bind": "/etc/bentoml_config.yml",
                "mode": "rw",
            }
        }
    else:
        volumes = None

    try:
        container = client.containers.run(
            image=image.id,
            command=command_args,
            auto_remove=True,
            tty=True,
            ports={'5000/tcp': port},
            detach=True,
            remove=True,
            volumes=volumes,
        )
    except Exception:  # pylint: disable=broad-except
        return

    try:
        host_url = f"127.0.0.1:{port}"
        _wait_until_api_server_ready(host_url, timeout, container)
        yield host_url
    finally:
        print(container.logs())
        container.stop()
        time.sleep(1)  # make sure container stopped & deleted


@contextmanager
def run_api_server(
    bundle_path, enable_microbatch=False, config_file=None, dev_server=False, timeout=20
):
    """
    Launch a bentoml service directly by the bentoml CLI, yields the host URL.
    """

    if dev_server:
        serve_cmd = "serve"
    else:
        serve_cmd = "serve-gunicorn"

    my_env = os.environ.copy()

    with bentoml.utils.reserve_free_port() as port:
        cmd = [sys.executable, "-m", "bentoml", serve_cmd]
        if port:
            cmd += ['--port', f'{port}']
        if enable_microbatch:
            cmd += ['--enable-microbatch']
        cmd += [bundle_path]

    def print_log(p):
        try:
            for line in p.stdout:
                print(line.decode(), end='')
        except ValueError:
            pass

    if config_file is not None:
        my_env["BENTOML_CONFIG"] = os.path.abspath(config_file)

    p = subprocess.Popen(
        cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, env=my_env,
    )
    try:
        threading.Thread(target=print_log, args=(p,), daemon=True).start()
        host_url = f"127.0.0.1:{port}"
        _wait_until_api_server_ready(host_url, timeout=timeout)
        yield host_url
    finally:
        # TODO: can not terminate the subprocess on Windows
        p.terminate()
        p.wait()
