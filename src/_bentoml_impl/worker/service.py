from __future__ import annotations

import json
import os
import typing as t

import click


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
    "--prometheus-dir",
    type=click.Path(exists=True),
    help="Required by prometheus to pass the metrics in multi-process mode",
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
def main(
    bento_identifier: str,
    service_name: str,
    fd: int,
    runner_map: str | None,
    backlog: int,
    worker_env: str | None,
    worker_id: int | None,
    prometheus_dir: str | None,
    ssl_certfile: str | None,
    ssl_keyfile: str | None,
    ssl_keyfile_password: str | None,
    ssl_version: int | None,
    ssl_cert_reqs: int | None,
    ssl_ca_certs: str | None,
    ssl_ciphers: str | None,
    development_mode: bool,
    timeout: int,
):
    """
    Start a HTTP server worker for given service.
    """
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

    from ..server.app import ServiceAppFactory

    if runner_map:
        BentoMLContainer.remote_runner_mapping.set(
            t.cast(t.Dict[str, str], json.loads(runner_map))
        )

    service = import_service(bento_identifier)
    service.inject_config()

    if service_name and service_name != service.name:
        service = service.find_dependent(service_name)
        server_context.service_type = "service"
    else:
        server_context.service_type = "entry_service"

    if worker_id is not None:
        server_context.worker_index = worker_id

    configure_server_logging()
    BentoMLContainer.development_mode.set(development_mode)

    if prometheus_dir is not None:
        BentoMLContainer.prometheus_multiproc_dir.set(prometheus_dir)
    server_context.service_name = service.name

    asgi_app = ServiceAppFactory(
        service, is_main=server_context.service_type == "entry_service"
    )()

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

    uvicorn.run(
        app=asgi_app,
        fd=fd,
        backlog=backlog,
        log_config=None,
        workers=1,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_keyfile_password=ssl_keyfile_password,
        ssl_ca_certs=ssl_ca_certs,
        **uvicorn_extra_options,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
