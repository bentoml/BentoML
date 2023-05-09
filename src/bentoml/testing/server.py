from __future__ import annotations

import os
import sys
import time
import socket
import typing as t
import urllib
import logging
import itertools
import contextlib
import subprocess
import urllib.error
import urllib.request
import multiprocessing
from contextlib import contextmanager

import yaml
import psutil

import bentoml
from bentoml.client import GrpcClient
from bentoml.client import HTTPClient
from bentoml.server import GrpcServer
from bentoml.server import HTTPServer
from bentoml._internal.utils import reserve_free_port
from bentoml._internal.utils import cached_contextmanager

from ..grpc.utils import import_grpc

if t.TYPE_CHECKING:
    import grpc
    from grpc import aio
else:
    grpc, aio = import_grpc()


logger = logging.getLogger(__name__)


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


def server_warmup(
    host: str,
    port: int,
    timeout: int,
    use_grpc: bool = False,
    check_interval: int = 1,
    popen: subprocess.Popen[t.Any] | None = None,
) -> bool:
    logger.info("Waiting for host %s:%d to be ready.." % (host, port))
    if use_grpc:
        try:
            GrpcClient.wait_until_server_ready(host, port, timeout, check_interval)
            return True
        except (TimeoutError, grpc.RpcError, aio.AioRpcError):
            return False
    else:
        if popen and popen.poll() is not None:
            return False
        else:
            try:
                HTTPClient.wait_until_server_ready(host, port, timeout, check_interval)
                return True
            except (
                ConnectionError,
                urllib.error.URLError,
                socket.timeout,
                ConnectionRefusedError,
                TimeoutError,
            ):
                return False


@cached_contextmanager("{project_path}, {cleanup}")
def build(
    project_path: str, cleanup: bool = True
) -> t.Generator[bentoml.Bento, None, None]:
    """
    Build a BentoML project.
    """
    from bentoml import bentos

    logger.info(f"Building bento from path: {project_path}")
    bento = bentos.build_bentofile(build_ctx=project_path)
    yield bento
    if cleanup:
        logger.info(f"Deleting bento: {str(bento.tag)}")
        bentos.delete(bento.tag)


@cached_contextmanager("{bento_tag}, {image_tag}, {cleanup}, {use_grpc}")
def containerize(
    bento_tag: str | bentoml.Tag,
    image_tag: str | None = None,
    cleanup: bool = True,
    use_grpc: bool = False,
    backend: str = "docker",
    **attrs: t.Any,
) -> t.Generator[str, None, None]:
    """
    Build the docker image from a saved bento, yield the docker image tag
    """
    from bentoml import container

    bento_tag = bentoml.Tag.from_taglike(bento_tag)
    if image_tag is None:
        image_tag = str(bento_tag)
    try:
        logger.info(f"Building bento server container: {bento_tag}")
        container.build(
            str(bento_tag),
            backend=backend,
            image_tag=(image_tag,),
            progress="plain",
            features=["grpc", "grpc-reflection"] if use_grpc else None,
            label={"testing": True, "module": __name__},
            **attrs,
        )
        yield image_tag
    finally:
        if cleanup:
            logger.info(f"Removing bento server container: {image_tag}")
            subprocess.call([backend, "rmi", image_tag])


@cached_contextmanager("{image_tag}, {config_file}, {use_grpc}, {platform}")
def run_bento_server_container(
    image_tag: str,
    config_file: str | None = None,
    use_grpc: bool = False,
    timeout: int = 90,
    host: str = "127.0.0.1",
    backend: str = "docker",
    platform: str = "linux/amd64",
):
    """
    Launch a bentoml service container from a container, yield the host URL
    """
    from bentoml._internal.configuration.containers import BentoMLContainer

    container_name = f"bentoml-test-{image_tag.replace(':', '_')}-{hash(config_file)}"
    with reserve_free_port(enable_so_reuseport=use_grpc) as port, reserve_free_port(
        enable_so_reuseport=use_grpc
    ) as prom_port:
        pass

    cmd = [
        backend,
        "run",
        "--rm",
        "--name",
        container_name,
        "--publish",
        f"{port}:3000",
        "--platform",
        platform,
    ]
    if config_file is not None:
        cmd.extend(["--env", "BENTOML_CONFIG=/home/bentoml/bentoml_config.yml"])
        cmd.extend(
            ["-v", f"{os.path.abspath(config_file)}:/home/bentoml/bentoml_config.yml"]
        )
    if use_grpc:
        cmd.extend(
            ["--publish", f"{prom_port}:{BentoMLContainer.grpc.metrics.port.get()}"]
        )
    cmd.append(image_tag)
    serve_cmd = "serve-grpc" if use_grpc else "serve-http"
    cmd.extend([serve_cmd])
    logger.info(f"Running API server in container: '{' '.join(cmd)}'")
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        encoding="utf-8",
    ) as proc:
        try:
            assert server_warmup(
                host, port, timeout=timeout, popen=proc, use_grpc=use_grpc
            )
            yield f"{host}:{port}"
        except Exception as e:
            raise RuntimeError(
                f"API server {host}:{port} failed to start within {timeout} seconds"
            ) from e
        finally:
            logger.info(f"Stopping Bento container {container_name}...")
            subprocess.call([backend, "stop", container_name])
    time.sleep(1)


@contextmanager
def run_bento_server_standalone(
    bento: str,
    use_grpc: bool = False,
    config_file: str | None = None,
    timeout: int = 90,
    host: str = "127.0.0.1",
):
    """
    Launch a bentoml service directly by the bentoml CLI, yields the host URL.
    """
    copied = os.environ.copy()

    if config_file is not None:
        copied["BENTOML_CONFIG"] = os.path.abspath(config_file)

    with reserve_free_port(host=host, enable_so_reuseport=use_grpc) as server_port:
        pass

    if use_grpc:
        server = GrpcServer(bento, production=True, host=host, port=server_port)
    else:
        server = HTTPServer(bento, production=True, host=host, port=server_port)
    server.timeout = timeout

    try:
        logger.info(f"Running command: '{' '.join(server.args)}'")
        server.start(env=copied)
        assert server_warmup(
            server.host, int(server.port), timeout=timeout, use_grpc=use_grpc
        )
        yield f"{server.host}:{server.port}"
    finally:
        server.stop()


def start_mitm_proxy(port: int) -> None:
    import uvicorn

    from .http import http_proxy_app

    logger.info(f"Proxy server listen on {port}")
    uvicorn.run(http_proxy_app, port=port)


@contextmanager
def run_bento_server_distributed(
    bento_tag: str | bentoml.Tag,
    config_file: str | None = None,
    use_grpc: bool = False,
    timeout: int = 90,
    host: str = "127.0.0.1",
):
    """
    Launch a bentoml service as a simulated distributed environment(Yatai), yields the host URL.
    """
    with reserve_free_port(enable_so_reuseport=use_grpc) as proxy_port:
        pass
    logger.info(f"Starting proxy on port {proxy_port}")
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
    bento = bentoml.bentos.get(bento_tag)
    with open(bento.path_of("bento.yaml"), "r", encoding="utf-8") as f:
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
                bento.path,
            ]
            logger.info(f"Running command: '{' '.join(cmd)}'")
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
        bento.path,
        *itertools.chain.from_iterable(runner_args),
    ]
    with reserve_free_port(host=host, enable_so_reuseport=use_grpc) as server_port:
        cmd.extend(["--port", f"{server_port}"])
    logger.info(f"Running command: '{' '.join(cmd)}'")
    processes.append(subprocess.Popen(cmd, env=copied))
    try:
        assert server_warmup(host, server_port, timeout=timeout, use_grpc=use_grpc)
        yield f"{host}:{server_port}"
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
    bento_name: str | bentoml.Tag | None = None,
    project_path: str = ".",
    config_file: str | None = None,
    deployment_mode: t.Literal["standalone", "distributed", "container"] = "standalone",
    bentoml_home: str | None = None,
    use_grpc: bool = False,
    clean_context: contextlib.ExitStack | None = None,
    host: str = "127.0.0.1",
    timeout: int = 120,
    backend: str = "docker",
    container_mode_options: dict[str, t.Any] | None = None,
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
        timeout: the timeout for the server to start
        backend: the backend to use for building container, default to ``docker``

    Returns:
        :obj:`str`: a generated host URL where we run the test bento.
    """
    import bentoml

    # NOTE: clean_context here to ensure we can close all running context manager
    # to avoid dangling processes. When running pytest, we have a clean_context fixture
    # as a ExitStack.
    if clean_context is None:
        clean_context = contextlib.ExitStack()
        clean_on_exit = True
    else:
        clean_on_exit = False

    # NOTE: we need to set the BENTOML_HOME to a temporary folder to avoid
    # conflict with the user's BENTOML_HOME.
    if bentoml_home:
        from bentoml._internal.configuration.containers import BentoMLContainer

        BentoMLContainer.bentoml_home.set(bentoml_home)

    try:
        if bento_name is None or not bentoml.list(bento_name):
            bento = clean_context.enter_context(
                build(project_path, cleanup=clean_on_exit)
            )
        else:
            bento = bentoml.get(bento_name)
        logger.info(
            f"Hosting BentoServer '{bento.tag}' in {deployment_mode} mode at '{project_path}'{' with config file '+config_file if config_file else ''}."
        )
        if deployment_mode == "standalone":
            with run_bento_server_standalone(
                bento.path,
                config_file=config_file,
                use_grpc=use_grpc,
                host=host,
                timeout=timeout,
            ) as host_url:
                yield host_url
        elif deployment_mode == "container":
            from bentoml._internal.container import REGISTERED_BACKENDS

            if container_mode_options is None:
                container_mode_options = {}

            if "platform" not in container_mode_options:
                # NOTE: by default, we use linux/amd64 for the container image
                container_mode_options["platform"] = "linux/amd64"

            platform = container_mode_options["platform"]
            if backend not in REGISTERED_BACKENDS:
                raise ValueError(
                    f"Unknown backend: {backend}. To register your custom backend, use 'bentoml.container.register_backend()'"
                )
            container_tag = clean_context.enter_context(
                containerize(
                    bento.tag,
                    use_grpc=use_grpc,
                    backend=backend,
                    **container_mode_options,
                )
            )
            with run_bento_server_container(
                container_tag,
                config_file=config_file,
                use_grpc=use_grpc,
                host=host,
                timeout=timeout,
                backend=backend,
                platform=platform,
            ) as host_url:
                yield host_url
        elif deployment_mode == "distributed":
            with run_bento_server_distributed(
                bento.tag,
                config_file=config_file,
                use_grpc=use_grpc,
                host=host,
                timeout=timeout,
            ) as host_url:
                yield host_url
        else:
            raise ValueError(f"Unknown deployment mode: {deployment_mode}") from None
    finally:
        logger.info("Shutting down bento server...")
        if clean_on_exit:
            logger.info("Cleaning on exit...")
            clean_context.close()
