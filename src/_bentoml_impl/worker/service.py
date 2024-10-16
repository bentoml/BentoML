from __future__ import annotations

import json
import logging
import os
import typing as t

import click

logger = logging.getLogger("bentoml.worker.service")


def patch_safetensor():
    import os
    import subprocess

    try:
        import safetensors.torch
    except ImportError:
        logger.info("safetensors not installed, skipping model preheat")
        return

    logger.info(
        "Patching safetensors.torch.safe_open to preheat model loading in parallel"
    )

    # Save the original safe_open class
    OriginalSafeOpen = safetensors.torch.safe_open

    # Define a new class to wrap around the original safe_open class
    class PatchedSafeOpen:
        def __init__(self, filename, framework, device="cpu"):
            # Call the read_ahead method before the usual safe_open
            self.read_ahead(filename)

            # Initialize the original safe_open
            self._original_safe_open = OriginalSafeOpen(filename, framework, device)

        def __enter__(self):
            return self._original_safe_open.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            return self._original_safe_open.__exit__(exc_type, exc_value, traceback)

        @staticmethod
        def read_ahead(
            file_path,
            num_processes=None,
            size_threshold=100 * 1024 * 1024,
            block_size=1024 * 1024,
        ):
            """
            Read a file in parallel using multiple processes.

            Args:
                file_path: Path to the file to read
                num_processes: Number of processes to use for reading the file. If None, the number of processes is set to the number of CPUs.
                size_threshold: If the file size is smaller than this threshold, only one process is used to read the file.
                block_size: Block size to use for reading the file
            """
            if num_processes is None:
                num_processes = os.cpu_count() or 8

            file_size = os.path.getsize(file_path)
            if file_size <= size_threshold:
                num_processes = 1

            chunk_size = file_size // num_processes
            processes = []

            for i in range(1, num_processes):
                start_byte = i * chunk_size
                end_byte = (
                    start_byte + chunk_size if i < num_processes - 1 else file_size
                )
                logger.info(
                    f"Reading bytes {start_byte} to {end_byte} from {file_path}"
                )
                process = subprocess.Popen(
                    [
                        "dd",
                        f"if={file_path}",
                        "of=/dev/null",
                        f"bs={block_size}",
                        f"skip={start_byte // block_size}",
                        f"count={(end_byte - start_byte) // block_size}",
                        "status=none",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                processes.append(process)

        def __getattr__(self, name):
            return getattr(self._original_safe_open, name)

    # Patch the original safetensors.torch module directly
    safetensors.torch.safe_open = PatchedSafeOpen


def get_model_preheat():
    """Get the model preheat flag from the environment variable BENTOML_MODEL_PREHEAT."""
    env_var = os.getenv("BENTOML_MODEL_PREHEAT", "0")
    return env_var not in ("0", "false", "False", "")


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--service-name", type=click.STRING, required=False, default="")
@click.option(
    "--fd",
    type=click.INT,
    required=True,
    help="File descriptor of the socket to listen on",
)
@click.option(
    "--runner-map",
    type=click.STRING,
    envvar="BENTOML_RUNNER_MAP",
    help="JSON string of runners map, default sets to envars `BENTOML_RUNNER_MAP`",
)
@click.option(
    "--backlog", type=click.INT, default=2048, help="Backlog size for the socket"
)
@click.option(
    "--worker-env", type=click.STRING, default=None, help="Environment variables"
)
@click.option(
    "--worker-id",
    required=False,
    type=click.INT,
    default=None,
    help="If set, start the server as a bare worker with the given worker ID. Otherwise start a standalone server with a supervisor process.",
)
@click.option(
    "--ssl-certfile",
    type=str,
    default=None,
    help="SSL certificate file",
)
@click.option(
    "--ssl-keyfile",
    type=str,
    default=None,
    help="SSL key file",
)
@click.option(
    "--ssl-keyfile-password",
    type=str,
    default=None,
    help="SSL keyfile password",
)
@click.option(
    "--ssl-version",
    type=int,
    default=None,
    help="SSL version to use (see stdlib 'ssl' module)",
)
@click.option(
    "--ssl-cert-reqs",
    type=int,
    default=None,
    help="Whether client certificate is required (see stdlib 'ssl' module)",
)
@click.option(
    "--ssl-ca-certs",
    type=str,
    default=None,
    help="CA certificates file",
)
@click.option(
    "--ssl-ciphers",
    type=str,
    default=None,
    help="Ciphers to use (see stdlib 'ssl' module)",
)
@click.option(
    "--timeout-keep-alive",
    type=int,
    default=5,
    help="Close Keep-Alive connections if no new data is received within this timeout.",
)
@click.option(
    "--timeout-graceful-shutdown",
    type=int,
    default=None,
    help="Maximum number of seconds to wait for graceful shutdown. After this timeout, the server will start terminating requests.",
)
@click.option(
    "--development-mode",
    type=click.BOOL,
    help="Run the API server in development mode",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "--timeout",
    type=click.INT,
    help="Specify the timeout for API server",
)
@click.option(
    "--model-preheat",
    is_flag=True,
    default=lambda: get_model_preheat(),
    help="Preheat model loading in parallel using multiple processes. Can be set via the environment variable BENTOML_MODEL_PREHEAT.",
)
def main(
    bento_identifier: str,
    service_name: str,
    fd: int,
    runner_map: str | None,
    backlog: int,
    worker_env: str | None,
    worker_id: int | None,
    ssl_certfile: str | None,
    ssl_keyfile: str | None,
    ssl_keyfile_password: str | None,
    ssl_version: int | None,
    ssl_cert_reqs: int | None,
    ssl_ca_certs: str | None,
    ssl_ciphers: str | None,
    timeout_keep_alive: int | None,
    timeout_graceful_shutdown: int | None,
    development_mode: bool,
    timeout: int,
    model_preheat: bool,
):
    """
    Start a HTTP server worker for given service.
    """
    import socket

    import psutil
    import uvicorn

    if worker_env:
        env_list: list[dict[str, t.Any]] = json.loads(worker_env)
        if worker_id is not None:
            # worker id from circus starts from 1
            worker_key = worker_id - 1
            if worker_key >= len(env_list):
                raise IndexError(
                    f"Worker ID {worker_id} is out of range, "
                    f"the maximum worker ID is {len(env_list)}"
                )
            os.environ.update(env_list[worker_key])

    from _bentoml_impl.loader import import_service
    from bentoml._internal.container import BentoMLContainer
    from bentoml._internal.context import server_context
    from bentoml._internal.log import configure_server_logging

    configure_server_logging()
    if runner_map:
        BentoMLContainer.remote_runner_mapping.set(
            t.cast(t.Dict[str, str], json.loads(runner_map))
        )

    if worker_id is not None:
        server_context.worker_index = worker_id
    service = import_service(bento_identifier)

    if service_name and service_name != service.name:
        service = service.find_dependent(service_name)
        server_context.service_type = "service"
    else:
        server_context.service_type = "entry_service"

    BentoMLContainer.development_mode.set(development_mode)

    server_context.service_name = service.name
    if service.bento is None:
        server_context.bento_name = service.name
        server_context.bento_version = "not available"
    else:
        server_context.bento_name = service.bento.tag.name
        server_context.bento_version = service.bento.tag.version or "not available"

    asgi_app = service.to_asgi(is_main=server_context.service_type == "entry_service")

    uvicorn_extra_options: dict[str, t.Any] = {}
    if ssl_version is not None:
        uvicorn_extra_options["ssl_version"] = ssl_version
    if ssl_cert_reqs is not None:
        uvicorn_extra_options["ssl_cert_reqs"] = ssl_cert_reqs
    if ssl_ciphers is not None:
        uvicorn_extra_options["ssl_ciphers"] = ssl_ciphers

    if psutil.WINDOWS:
        # 1. uvloop is not supported on Windows
        # 2. the default policy for Python > 3.8 on Windows is ProactorEventLoop, which doesn't
        #    support listen on a existing socket file descriptors
        # See https://docs.python.org/3.8/library/asyncio-platforms.html#windows
        uvicorn_extra_options["loop"] = "asyncio"
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

    config = uvicorn.Config(
        app=asgi_app,
        backlog=backlog,
        log_config=None,
        workers=1,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_ca_certs=ssl_ca_certs,
        timeout_keep_alive=timeout_keep_alive,
        timeout_graceful_shutdown=timeout_graceful_shutdown,
        server_header=False,
        **uvicorn_extra_options,
    )
    socket = socket.socket(fileno=fd)
    if model_preheat:
        patch_safetensor()

    uvicorn.Server(config).run(sockets=[socket])


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
