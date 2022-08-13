from __future__ import annotations

import sys
import json
import socket
import typing as t

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option(
    "--bind",
    type=click.STRING,
    required=True,
    help="Bind address sent to circus. This address accepts the following values: 'tcp://127.0.0.1:3000','unix:///tmp/bento_api.sock', 'fd://12'",
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
    "--working-dir",
    type=click.Path(exists=True),
    help="Working directory for the API server",
)
@click.option(
    "--prometheus-dir",
    type=click.Path(exists=True),
    help="Required by prometheus to pass the metrics in multi-process mode",
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
@click.pass_context
def main(
    ctx: click.Context,
    bento_identifier: str,
    bind: str,
    runner_map: str | None,
    backlog: int,
    working_dir: str | None,
    worker_id: int | None,
    prometheus_dir: str | None,
    ssl_certfile: str | None,
    ssl_keyfile: str | None,
    ssl_keyfile_password: str | None,
    ssl_version: int | None,
    ssl_cert_reqs: int | None,
    ssl_ca_certs: str | None,
    ssl_ciphers: str | None,
):
    """
    Start BentoML API server.
    \b
    This is an internal API, users should not use this directly. Instead use `bentoml serve <path> [--options]`
    """

    from urllib.parse import urlparse

    import psutil

    import bentoml
    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.context import component_context
    from bentoml._internal.configuration.containers import BentoMLContainer

    configure_server_logging()

    BentoMLContainer.development_mode.set(False)
    if prometheus_dir is not None:
        BentoMLContainer.prometheus_multiproc_dir.set(prometheus_dir)

    if worker_id is None:
        # Start a standalone server with a supervisor process
        from circus.watcher import Watcher

        from bentoml.serve import ensure_prometheus_dir
        from bentoml_cli.utils import unparse_click_params
        from bentoml._internal.utils.circus import create_standalone_arbiter
        from bentoml._internal.utils.circus import create_circus_socket_from_uri

        ensure_prometheus_dir()

        circus_socket = create_circus_socket_from_uri(bind, name="_bento_api_server")
        params = ctx.params
        params["bind"] = "fd://$(circus.sockets._bento_api_server)"
        params["worker_id"] = "$(circus.wid)"
        watcher = Watcher(
            name="bento_api_server",
            cmd=sys.executable,
            args=["-m", "bentoml_cli.server.http_api_server"]
            + unparse_click_params(params, ctx.command.params, factory=str),
            copy_env=True,
            numprocesses=1,
            stop_children=True,
            use_sockets=True,
            working_dir=working_dir,
        )
        arbiter = create_standalone_arbiter(watchers=[watcher], sockets=[circus_socket])
        arbiter.start()
        return

    component_context.component_name = f"api_server:{worker_id}"

    if runner_map is not None:
        BentoMLContainer.remote_runner_mapping.set(json.loads(runner_map))
    svc = bentoml.load(bento_identifier, working_dir=working_dir, standalone_load=True)

    # setup context
    if svc.tag is None:
        component_context.bento_name = f"*{svc.__class__.__name__}"
        component_context.bento_version = "not available"
    else:
        component_context.bento_name = svc.tag.name
        component_context.bento_version = svc.tag.version

    parsed = urlparse(bind)
    uvicorn_options: dict[str, t.Any] = {
        "backlog": backlog,
        "log_config": None,
        "workers": 1,
    }

    if ssl_certfile:
        import ssl

        uvicorn_options["ssl_certfile"] = ssl_certfile
        if ssl_keyfile:
            uvicorn_options["ssl_keyfile"] = ssl_keyfile
        if ssl_keyfile_password:
            uvicorn_options["ssl_keyfile_password"] = ssl_keyfile_password
        if ssl_ca_certs:
            uvicorn_options["ssl_ca_certs"] = ssl_ca_certs

        if not ssl_version:
            ssl_version = ssl.PROTOCOL_TLS_SERVER
            uvicorn_options["ssl_version"] = ssl_version
        if not ssl_cert_reqs:
            ssl_cert_reqs = ssl.CERT_NONE
            uvicorn_options["ssl_cert_reqs"] = ssl_cert_reqs
        if not ssl_ciphers:
            ssl_ciphers = "TLSv1"
            uvicorn_options["ssl_ciphers"] = ssl_ciphers

    if psutil.WINDOWS:
        uvicorn_options["loop"] = "asyncio"
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

    assert parsed.scheme == "fd"

    import uvicorn

    # skip the uvicorn internal supervisor
    fd = int(parsed.netloc)
    sock = socket.socket(fileno=fd)
    config = uvicorn.Config(svc.asgi_app, **uvicorn_options)
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
