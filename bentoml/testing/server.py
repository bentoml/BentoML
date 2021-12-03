# pylint: disable=redefined-outer-name # pragma: no cover
import os
import sys
import time
import typing as t
import urllib
import logging
import threading
import contextlib
import subprocess
import urllib.request
from typing import TYPE_CHECKING
from contextlib import contextmanager

from .._internal.utils import reserve_free_port
from .._internal.utils import cached_contextmanager

logger = logging.getLogger("bentoml.tests")


if TYPE_CHECKING:
    from aiohttp.typedefs import LooseHeaders
    from starlette.datastructures import Headers
    from starlette.datastructures import FormData

    import docker


clean_context = contextlib.ExitStack()


@cached_contextmanager("{bento}, {config_file}, {workdir}, {docker}, {dev_server}")
def host_bento(
    bento: str,
    config_file: str = "bentoml_config.yml",
    workdir: str = "./",
    docker: bool = False,
    dev_server: bool = False,
) -> t.Generator[str, None, None]:
    # TODO: currently not used

    if not os.path.exists(config_file):
        raise Exception(f"config file not found: {config_file}")

    if docker:
        image = clean_context.enter_context(
            build_api_server_docker_image(bento, "example_service")  # TODO
        )
        host = clean_context.enter_context(
            run_api_server_in_docker(
                image,
                config_file=config_file,
            )
        )
        yield host
    elif dev_server:
        host = clean_context.enter_context(
            run_api_server(
                bento,
                config_file=config_file,
                workdir=workdir,
                dev_server=True,
            )
        )
        yield host
    else:
        host = clean_context.enter_context(
            run_api_server(
                bento,
                config_file=config_file,
                workdir=workdir,
            )
        )
        yield host


async def parse_multipart_form(headers: "Headers", body: bytes) -> "FormData":
    """
    parse starlette forms from headers and body
    """

    from starlette.formparsers import MultiPartParser

    async def async_bytesio(bytes_: bytes) -> t.AsyncGenerator[bytes, None]:
        yield bytes_
        yield b""
        return

    parser = MultiPartParser(headers=headers, stream=async_bytesio(body))
    return await parser.parse()


async def async_request(
    method: str,
    url: str,
    headers: t.Optional["LooseHeaders"] = None,
    data: t.Any = None,
    timeout: t.Optional[int] = None,
) -> t.Tuple[int, "Headers", bytes]:
    """
    raw async request client
    """
    import aiohttp
    from starlette.datastructures import Headers

    async with aiohttp.ClientSession() as sess:
        async with sess.request(
            method, url, data=data, headers=headers, timeout=timeout
        ) as r:
            r_body = await r.read()

    headers = t.cast(t.Mapping[str, str], r.headers)
    return r.status, Headers(headers), r_body


def wait_until_container_ready(container_name, check_message, timeout_seconds=120):
    import docker
    import docker.errors

    docker_client = docker.from_env()

    start_time = time.time()
    while True:
        time.sleep(1)

        # Raise timeout, if exceeds timeout limit
        if time.time() - start_time > timeout_seconds:
            raise TimeoutError(f'Waiting for container "{container_name}" timed out')

        try:
            container_list = docker_client.containers.list(
                filters={"name": container_name}
            )
            if not container_list:
                continue
        except docker.errors.NotFound:
            continue

        logger.info("Container list: " + str(container_list))
        assert (
            len(container_list) == 1
        ), f"should be exact one container with name {container_name}"

        container_log = container_list[0].logs().decode()
        if check_message in container_log:
            logger.info(
                f"Found message indicating container readiness in container log: "
                f"{container_log}"
            )
            break


def _wait_until_api_server_ready(host_url, timeout, container=None, check_interval=1):
    start_time = time.time()
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    ex = None
    while time.time() - start_time < timeout:
        try:
            if opener.open(f"http://{host_url}/readyz", timeout=1).status == 200:
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
                    for log_record in container_logs.decode().split("\r\n"):
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
def build_api_server_docker_image(
    saved_bundle_path, image_tag="test_bentoml_server"
) -> "docker.Image":
    """
    Build the docker image for a saved bentoml bundle, yield the docker image object.
    """

    import docker
    import docker.errors

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
            if "stream" in line:
                print(line["stream"].strip())
        raise


@cached_contextmanager("{image.id}")
def run_api_server_in_docker(image, config_file=None, timeout=90):
    """
    Launch a bentoml service container from a docker image, yields the host URL.
    """
    import docker

    client = docker.from_env()

    with reserve_free_port() as port:
        pass

    command_args = "--workers 1"

    if config_file is not None:
        environment = ["BENTOML_CONFIG=/home/bentoml/bentoml_config.yml"]
        volumes = {
            os.path.abspath(config_file): {
                "bind": "/home/bentoml/bentoml_config.yml",
                "mode": "ro",
            }
        }
    else:
        environment = None
        volumes = None

    container = client.containers.run(
        image=image.id,
        command=command_args,
        tty=True,
        ports={"5000/tcp": port},
        detach=True,
        volumes=volumes,
        environment=environment,
    )

    try:
        host_url = f"127.0.0.1:{port}"
        _wait_until_api_server_ready(host_url, timeout, container)
        yield host_url
    finally:
        print(container.logs())
        container.stop()
        container.remove()
        time.sleep(1)  # make sure container stopped & deleted


@contextmanager
def run_api_server(
    bento,
    workdir=None,
    config_file=None,
    dev_server=False,
    timeout=90,
):
    """
    Launch a bentoml service directly by the bentoml CLI, yields the host URL.
    """
    workdir = workdir if workdir is not None else "./"

    serve_cmd = "serve"

    my_env = os.environ.copy()

    with reserve_free_port() as port:
        cmd = [sys.executable, "-m", "bentoml", serve_cmd]

        if not dev_server:
            cmd += ["--production"]

        if port:
            cmd += ["--port", f"{port}"]
        cmd += [bento]
        cmd += ["--working-dir", workdir]

    print(cmd)

    def print_log(p):
        try:
            for line in p.stdout:
                print(line.decode(), end="")
        except ValueError:
            pass

    if config_file is not None:
        my_env["BENTOML_CONFIG"] = os.path.abspath(config_file)

    p = subprocess.Popen(
        cmd,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        env=my_env,
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
