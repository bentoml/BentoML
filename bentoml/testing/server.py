# pylint: disable=redefined-outer-name # pragma: no cover
import os
import sys
import time
import socket
import typing as t
import urllib
import logging
import contextlib
import subprocess
import urllib.error
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


clean_context = contextlib.ExitStack()


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


def _wait_until_api_server_ready(
    host_url: str,
    timeout: float,
    check_interval: float = 1,
    popen: t.Optional["subprocess.Popen[bytes]"] = None,
) -> bool:
    start_time = time.time()
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)

    logger.info("Waiting for host %s to be ready..", host_url)
    while time.time() - start_time < timeout:
        try:
            if popen and popen.poll() is not None:
                return False
            elif opener.open(f"http://{host_url}/readyz", timeout=1).status == 200:
                return True
            else:
                time.sleep(check_interval)
        except (
            ConnectionError,
            urllib.error.URLError,
            socket.timeout,
        ) as e:
            logger.info(f"[{e}]retrying to connect to the host {host_url}...")
            logger.error(e)
            time.sleep(check_interval)
    else:
        logger.info(
            f"Timed out waiting {timeout} seconds for Server {host_url} to be ready, "
        )
        return False


@contextmanager
def export_service_bundle(_):
    """
    Export a bentoml service to a temporary directory, yield the path.
    Delete the temporary directory on close.
    """
    yield


@cached_contextmanager("{project_path}, {image_tag}")
def build_api_server_docker_image(
    project_path: str, image_tag: t.Optional[str] = None
) -> t.Generator[str, None, None]:
    """
    Build the docker image for a saved bentoml bundle, yield the docker image object.
    """
    import bentoml

    bento = bentoml.bentos.build_bentofile(build_ctx=project_path)
    tag = bento.info.tag
    if image_tag is None:
        image_tag = tag.name

    logger.info(f"Building API server docker image from build context: {project_path}")
    subprocess.check_call(["bentoml", "containerize", str(tag), "-t", image_tag])

    yield image_tag
    subprocess.call(["docker", "rmi", image_tag])


@cached_contextmanager("{image_tag}, {config_file}")
def run_api_server_in_docker(
    image_tag: str,
    config_file: t.Optional[str] = None,
    timeout: float = 40,
):
    """
    Launch a bentoml service container from a docker image, yields the host URL.
    """
    container_name = f"bentoml-test-{image_tag}"
    with reserve_free_port() as port:
        pass

    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--publish",
        f"{port}:3000",
        "--env",
        "BENTOML_LOG_STDOUT=true",
        "--env",
        "BENTOML_LOG_STDERR=true",
    ]

    if config_file is not None:
        cmd.extend(["--env", "BENTOML_CONFIG=/home/bentoml/bentoml_config.yml"])
        cmd.extend(
            ["-v", f"{os.path.abspath(config_file)}:/home/bentoml/bentoml_config.yml"]
        )
    cmd.append(image_tag)

    logger.info(f"Running API server docker image: {cmd}")
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        try:
            host_url = f"127.0.0.1:{port}"
            if _wait_until_api_server_ready(host_url, timeout, popen=proc):
                yield host_url
            else:
                raise RuntimeError(
                    f"API server {host_url} failed to start within {timeout} seconds"
                )
        finally:
            proc.terminate()
    time.sleep(1)


@contextmanager
def run_api_server(
    bento: str,
    workdir: t.Optional[str] = None,
    config_file: t.Optional[str] = None,
    dev_server: bool = False,
    timeout: float = 90,
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

    # def print_log(p: "subprocess.Popen"):
    # try:
    # for line in p.stdout:
    # print(line.decode(), end="")
    # except ValueError:
    # pass

    if config_file is not None:
        my_env["BENTOML_CONFIG"] = os.path.abspath(config_file)

    p = subprocess.Popen(
        cmd,
        stderr=subprocess.STDOUT,
        env=my_env,
    )
    try:
        # threading.Thread(target=print_log, args=(p,), daemon=True).start()
        host_url = f"127.0.0.1:{port}"
        _wait_until_api_server_ready(host_url, timeout=timeout)
        yield host_url
    finally:
        # TODO: can not terminate the subprocess on Windows
        p.terminate()
        p.wait()


@cached_contextmanager("{bento}, {config_file}, {workdir}, {docker}, {dev_server}")
def host_bento(
    bento: str,
    config_file: str = "bentoml_config.yml",
    workdir: str = "./",
    docker: bool = False,
    dev_server: bool = False,
) -> t.Generator[str, None, None]:

    if not os.path.exists(config_file):
        raise Exception(f"config file not found: {config_file}")

    try:
        if docker:
            logger.info("Building API server docker image...")
            with build_api_server_docker_image(workdir) as image_tag:
                logger.info("Running API server docker image...")
                with run_api_server_in_docker(image_tag, config_file) as host_url:
                    yield host_url
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
    finally:
        logger.info("Cleaning up...")
