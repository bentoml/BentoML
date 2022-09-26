# pylint: disable=redefined-outer-name,not-context-manager
from __future__ import annotations

import os
import sys
import time
import socket
import typing as t
import urllib
import asyncio
import itertools
import contextlib
import subprocess
import urllib.error
import urllib.request
import multiprocessing
from typing import TYPE_CHECKING
from contextlib import contextmanager

import psutil

from bentoml.grpc.utils import import_grpc
from bentoml._internal.tag import Tag
from bentoml._internal.utils import LazyLoader
from bentoml._internal.utils import reserve_free_port
from bentoml._internal.utils import cached_contextmanager

if TYPE_CHECKING:
    from grpc import aio
    from grpc_health.v1 import health_pb2 as pb_health
    from aiohttp.typedefs import LooseHeaders
    from starlette.datastructures import Headers
    from starlette.datastructures import FormData

    from bentoml._internal.bento.bento import Bento

else:
    pb_health = LazyLoader("pb_health", globals(), "grpc_health.v1.health_pb2")
    _, aio = import_grpc()


async def parse_multipart_form(headers: Headers, body: bytes) -> FormData:
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
    headers: LooseHeaders | None = None,
    data: t.Any = None,
    timeout: int | None = None,
) -> tuple[int, Headers, bytes]:
    """
    A HTTP client with async API.
    """
    import aiohttp
    from starlette.datastructures import Headers

    async with aiohttp.ClientSession() as sess:
        async with sess.request(
            method,
            url,
            data=data,
            headers=headers,
            timeout=timeout,
        ) as r:
            r_body = await r.read()

    headers = t.cast(t.Mapping[str, str], r.headers)
    return r.status, Headers(headers), r_body


def kill_subprocess_tree(p: subprocess.Popen[t.Any]) -> None:
    """
    Tell the process to terminate and kill all of its children. Availabe both on Windows and Linux.
    Note: It will return immediately rather than wait for the process to terminate.

    Args:
        p: subprocess.Popen object
    """
    if psutil.WINDOWS:
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)])
    else:
        p.terminate()


async def server_warmup(
    host_url: str,
    timeout: float,
    grpc: bool = False,
    check_interval: float = 1,
    popen: subprocess.Popen[t.Any] | None = None,
    service_name: str | None = None,
) -> bool:
    from bentoml.testing.grpc import create_channel

    start_time = time.time()
    print("Waiting for host %s to be ready.." % host_url)
    while time.time() - start_time < timeout:
        try:
            if popen and popen.poll() is not None:
                return False
            elif grpc:
                if service_name is None:
                    service_name = "bentoml.grpc.v1alpha1.BentoService"
                async with create_channel(host_url) as channel:
                    Check = channel.unary_unary(
                        "/grpc.health.v1.Health/Check",
                        request_serializer=pb_health.HealthCheckRequest.SerializeToString,  # type: ignore (no grpc_health type)
                        response_deserializer=pb_health.HealthCheckResponse.FromString,  # type: ignore (no grpc_health type)
                    )
                    resp = await t.cast(
                        t.Awaitable[pb_health.HealthCheckResponse],
                        Check(
                            pb_health.HealthCheckRequest(service=service_name),
                            timeout=timeout,
                        ),
                    )
                    if resp.status == pb_health.HealthCheckResponse.SERVING:  # type: ignore (no generated enum types)
                        return True
                    else:
                        time.sleep(check_interval)
            else:
                proxy_handler = urllib.request.ProxyHandler({})
                opener = urllib.request.build_opener(proxy_handler)
                if opener.open(f"http://{host_url}/readyz", timeout=1).status == 200:
                    return True
                else:
                    time.sleep(check_interval)
        except (
            aio.AioRpcError,
            ConnectionError,
            urllib.error.URLError,
            socket.timeout,
        ) as e:
            print(f"[{e}] Retrying to connect to the host {host_url}...")
            time.sleep(check_interval)
    print(f"Timed out waiting {timeout} seconds for Server {host_url} to be ready.")
    return False


@cached_contextmanager("{project_path}, {cleanup}")
def bentoml_build(
    project_path: str, cleanup: bool = True
) -> t.Generator[Bento, None, None]:
    """
    Build a BentoML project.
    """
    from bentoml import bentos

    print(f"Building bento: {project_path}")
    bento = bentos.build_bentofile(build_ctx=project_path)
    yield bento
    if cleanup:
        print(f"Deleting bento: {str(bento.tag)}")
        bentos.delete(bento.tag)


@cached_contextmanager("{bento_tag}, {image_tag}, {cleanup}, {use_grpc}")
def bentoml_containerize(
    bento_tag: str | Tag,
    image_tag: str | None = None,
    cleanup: bool = True,
    use_grpc: bool = False,
) -> t.Generator[str, None, None]:
    """
    Build the docker image from a saved bento, yield the docker image tag
    """
    from bentoml import bentos

    bento_tag = Tag.from_taglike(bento_tag)
    if image_tag is None:
        image_tag = bento_tag.name
    try:
        print(f"Building bento server docker image: {bento_tag}")
        bentos.containerize(
            str(bento_tag),
            docker_image_tag=[image_tag],
            progress="plain",
            features=["grpc"] if use_grpc else None,
        )
        yield image_tag
    finally:
        if cleanup:
            print(f"Removing bento server docker image: {image_tag}")
            subprocess.call(["docker", "rmi", image_tag])


@cached_contextmanager("{image_tag}, {config_file}, {use_grpc}")
def run_bento_server_docker(
    image_tag: str,
    config_file: str | None = None,
    use_grpc: bool = False,
    timeout: float = 90,
    host: str = "127.0.0.1",
):
    """
    Launch a bentoml service container from a docker image, yield the host URL
    """
    from bentoml._internal.configuration.containers import BentoMLContainer

    container_name = f"bentoml-test-{image_tag}-{hash(config_file)}"
    with reserve_free_port(enable_so_reuseport=use_grpc) as port:
        pass
    bind_port = "3000"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--publish",
        f"{port}:{bind_port}",
    ]
    if config_file is not None:
        cmd.extend(["--env", "BENTOML_CONFIG=/home/bentoml/bentoml_config.yml"])
        cmd.extend(
            ["-v", f"{os.path.abspath(config_file)}:/home/bentoml/bentoml_config.yml"]
        )
    if use_grpc:
        bind_prom_port = BentoMLContainer.grpc.metrics.port.get()
        cmd.extend(["--publish", f"{bind_prom_port}:{bind_prom_port}"])
    cmd.append(image_tag)
    if use_grpc:
        cmd.extend(["serve-grpc", "--production", "--enable-reflection"])
    print(f"Running API server docker image: '{' '.join(cmd)}'")
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        encoding="utf-8",
    ) as proc:
        try:
            host_url = f"{host}:{port}"
            if asyncio.run(
                server_warmup(host_url, timeout=timeout, popen=proc, grpc=use_grpc)
            ):
                yield host_url
            else:
                raise RuntimeError(
                    f"API server {host_url} failed to start within {timeout} seconds"
                ) from None
        finally:
            print(f"Stopping Bento container {container_name}...")
            subprocess.call(["docker", "stop", container_name])
    time.sleep(1)


@contextmanager
def run_bento_server_standalone(
    bento: str,
    use_grpc: bool = False,
    config_file: str | None = None,
    timeout: float = 90,
    host: str = "127.0.0.1",
):
    """
    Launch a bentoml service directly by the bentoml CLI, yields the host URL.
    """
    copied = os.environ.copy()
    if config_file is not None:
        copied["BENTOML_CONFIG"] = os.path.abspath(config_file)
    with reserve_free_port(host=host, enable_so_reuseport=use_grpc) as server_port:
        cmd = [
            sys.executable,
            "-m",
            "bentoml",
            "serve-grpc" if use_grpc else "serve",
            "--production",
            "--port",
            f"{server_port}",
        ]
        if use_grpc:
            cmd += ["--host", f"{host}", "--enable-reflection"]
    cmd += [bento]
    print(f"Running command: '{' '.join(cmd)}'")
    p = subprocess.Popen(
        cmd,
        stderr=subprocess.STDOUT,
        env=copied,
        encoding="utf-8",
    )
    try:
        host_url = f"{host}:{server_port}"
        assert asyncio.run(
            server_warmup(host_url, timeout=timeout, popen=p, grpc=use_grpc)
        )
        yield host_url
    finally:
        print(f"Stopping process [{p.pid}]...")
        kill_subprocess_tree(p)
        p.communicate()


def start_mitm_proxy(port: int) -> None:
    import uvicorn

    from .utils import http_proxy_app

    print(f"Proxy server listen on {port}")
    uvicorn.run(http_proxy_app, port=port)  # type: ignore (not using ASGI3Application)


@contextmanager
def run_bento_server_distributed(
    bento_tag: str | Tag,
    config_file: str | None = None,
    use_grpc: bool = False,
    timeout: float = 90,
    host: str = "127.0.0.1",
):
    """
    Launch a bentoml service as a simulated distributed environment(Yatai), yields the host URL.
    """
    import yaml

    import bentoml

    with reserve_free_port(enable_so_reuseport=use_grpc) as proxy_port:
        pass
    print(f"Starting proxy on port {proxy_port}")
    proxy_process = multiprocessing.Process(
        target=start_mitm_proxy,
        args=(proxy_port,),
    )
    proxy_process.start()
    copied = os.environ.copy()
    # to ensure yatai specified headers BP100
    copied["YATAI_BENTO_DEPLOYMENT_NAME"] = "test-deployment"
    copied["YATAI_BENTO_DEPLOYMENT_NAMESPACE"] = "yatai"
    if use_grpc:
        copied["GPRC_PROXY"] = f"localhost:{proxy_port}"
    else:
        copied["HTTP_PROXY"] = f"http://127.0.0.1:{proxy_port}"
    if config_file is not None:
        copied["BENTOML_CONFIG"] = os.path.abspath(config_file)

    runner_map = {}
    processes: list[subprocess.Popen[str]] = []
    bento_service = bentoml.bentos.get(bento_tag)
    path = bento_service.path
    with open(os.path.join(path, "bento.yaml"), "r", encoding="utf-8") as f:
        bentofile = yaml.safe_load(f)
    for runner in bentofile["runners"]:
        with reserve_free_port(enable_so_reuseport=use_grpc) as port:
            runner_map[runner["name"]] = f"tcp://127.0.0.1:{port}"
            cmd = [
                sys.executable,
                "-m",
                "bentoml",
                "start-runner-server",
                str(bento_tag),
                "--runner-name",
                runner["name"],
                "--host",
                host,
                "--port",
                f"{port}",
                "--working-dir",
                path,
            ]
            print(f"Running command: '{' '.join(cmd)}'")
        processes.append(
            subprocess.Popen(
                cmd,
                encoding="utf-8",
                stderr=subprocess.STDOUT,
                env=copied,
            )
        )
    runner_args = [
        ("--remote-runner", f"{runner['name']}={runner_map[runner['name']]}")
        for runner in bentofile["runners"]
    ]
    cmd = [
        sys.executable,
        "-m",
        "bentoml",
        "start-http-server" if not use_grpc else "start-grpc-server",
        str(bento_tag),
        "--host",
        host,
        "--working-dir",
        path,
        *itertools.chain.from_iterable(runner_args),
    ]
    with reserve_free_port(host=host, enable_so_reuseport=use_grpc) as server_port:
        cmd.extend(["--port", f"{server_port}"])
        if use_grpc:
            cmd.append("--enable-reflection")
    print(f"Running command: '{' '.join(cmd)}'")
    processes.append(
        subprocess.Popen(
            cmd,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            env=copied,
        )
    )
    try:
        host_url = f"{host}:{server_port}"
        asyncio.run(server_warmup(host_url, timeout=timeout, grpc=use_grpc))
        yield host_url
    finally:
        for p in processes:
            kill_subprocess_tree(p)
        for p in processes:
            p.communicate()
        if proxy_process is not None:
            proxy_process.terminate()
            proxy_process.join()


@cached_contextmanager(
    "{bento_name}, {project_path}, {config_file}, {deployment_mode}, {bentoml_home}, {use_grpc}"
)
def host_bento(
    bento_name: str | Tag | None = None,
    project_path: str = ".",
    config_file: str | None = None,
    deployment_mode: t.Literal["standalone", "distributed", "docker"] = "standalone",
    bentoml_home: str | None = None,
    use_grpc: bool = False,
    clean_context: contextlib.ExitStack | None = None,
    host: str = "127.0.0.1",
) -> t.Generator[str, None, None]:
    """
    Host a bentoml service, yields the host URL.

    Args:
        bento: a bento tag or :code:`module_path:service`
        project_path: the path to the project directory
        config_file: the path to the config file
        deployment_mode: the deployment mode, one of :code:`standalone`, :code:`docker` or :code:`distributed`
        clean_context: a contextlib.ExitStack to clean up the intermediate files,
                       like docker image and bentos. If None, it will be created. Used for reusing
                       those files in the same test session.
        bentoml_home: if set, we will change the given BentoML home folder to :code:`bentoml_home`. Default
                      to :code:`$HOME/bentoml`
        use_grpc: if True, running gRPC tests.
        host: set a given host for the bento, default to ``127.0.0.1``

    Returns:
        :obj:`str`: a generated host URL where we run the test bento.
    """
    import bentoml

    if clean_context is None:
        clean_context = contextlib.ExitStack()
        clean_on_exit = True
    else:
        clean_on_exit = False
    if bentoml_home:
        from bentoml._internal.configuration.containers import BentoMLContainer

        BentoMLContainer.bentoml_home.set(bentoml_home)
    try:
        print(
            f"Starting bento server {bento_name} at '{project_path}' {'with config file '+config_file+' ' if config_file else ' '}in {deployment_mode} mode..."
        )
        if bento_name is None or not bentoml.list(bento_name):
            bento = clean_context.enter_context(bentoml_build(project_path))
        else:
            bento = bentoml.get(bento_name)
        if deployment_mode == "standalone":
            with run_bento_server_standalone(
                bento.path,
                config_file=config_file,
                use_grpc=use_grpc,
                host=host,
            ) as host_url:
                yield host_url
        elif deployment_mode == "docker":
            container_tag = clean_context.enter_context(
                bentoml_containerize(bento.tag, use_grpc=use_grpc)
            )
            with run_bento_server_docker(
                container_tag,
                config_file=config_file,
                use_grpc=use_grpc,
                host=host,
            ) as host_url:
                yield host_url
        elif deployment_mode == "distributed":
            with run_bento_server_distributed(
                bento.tag,
                config_file=config_file,
                use_grpc=use_grpc,
                host=host,
            ) as host_url:
                yield host_url
        else:
            raise ValueError(f"Unknown deployment mode: {deployment_mode}") from None
    finally:
        print("Shutting down bento server...")
        if clean_on_exit:
            print("Cleaning on exit...")
            clean_context.close()
