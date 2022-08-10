from __future__ import annotations

import socket

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--bind", type=click.STRING, required=True)
@click.option("--working-dir", required=False, type=click.Path(), default=None)
@click.option("--backlog", type=click.INT, default=2048)
@click.option(
    "--prometheus-dir",
    type=click.Path(exists=True),
    help="Required by prometheus to pass the metrics in multi-process mode",
)
@click.option(
    "--ssl-keyfile",
    type=click.STRING,
    help="SSL key file",
    default=None,
)
@click.option(
    "--ssl-certfile",
    type=click.STRING,
    help="SSL certificate file",
    default=None,
)
@click.option(
    "--ssl-keyfile-password",
    type=click.STRING,
    help="SSL keyfile password",
    default=None,
)
@click.option(
    "--ssl-version",
    type=click.INT,
    help="SSL version to use (see stdlib ssl module's)",
    default=None,
    # default=17 # TODO: default here, or set default to None and allow uvicorn to handle default?
)
@click.option(
    "--ssl-cert-reqs",
    type=click.INT,
    help="Whether client certificate is required (see stdlib ssl module's)",
    default=None,
    # default=0 # TODO: default here, or set default to None and allow uvicorn to handle default?
)
@click.option(
    "--ssl-ca-certs",
    type=click.STRING,
    help="CA certificates file",
    default=None,
)
@click.option(
    "--ssl-ciphers",
    type=click.STRING,
    help="CA certificates file",
    default=None,
    # default="TLSv1" # TODO: default here, or set default to None and allow uvicorn to handle default?
)
def main(
    bento_identifier: str,
    bind: str,
    working_dir: str | None,
    backlog: int,
    prometheus_dir: str | None,
    ssl_keyfile: str | None,
    ssl_certfile: str | None,
    ssl_keyfile_password: str | None,
    ssl_version: int | None,
    ssl_cert_reqs: int | None,
    ssl_ca_certs: str | None,
    ssl_ciphers: str | None,
):

    from urllib.parse import urlparse

    import psutil
    import uvicorn

    from bentoml import load
    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.context import component_context
    from bentoml._internal.configuration.containers import BentoMLContainer

    component_context.component_name = "dev_api_server"

    configure_server_logging()

    if prometheus_dir is not None:
        BentoMLContainer.prometheus_multiproc_dir.set(prometheus_dir)

    svc = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    # setup context
    if svc.tag is None:
        component_context.bento_name = f"*{svc.__class__.__name__}"
        component_context.bento_version = "not available"
    else:
        component_context.bento_name = svc.tag.name
        component_context.bento_version = svc.tag.version

    parsed = urlparse(bind)

    if parsed.scheme == "fd":
        fd = int(parsed.netloc)
        sock = socket.socket(fileno=fd)

        uvicorn_options = {
            "backlog": backlog,
            "log_config": None,
            "workers": 1,
            "lifespan": "on",
        }
        # Add optional SSL args if they exist
        if ssl_keyfile:
            uvicorn_options["ssl_keyfile"] = ssl_keyfile
        if ssl_certfile:
            uvicorn_options["ssl_certfile"] = ssl_certfile
        if ssl_keyfile_password:
            uvicorn_options["ssl_keyfile_password"] = ssl_keyfile_password
        if ssl_version:
            uvicorn_options["ssl_version"] = ssl_version
        if ssl_cert_reqs:
            uvicorn_options["ssl_cert_reqs"] = ssl_cert_reqs
        if ssl_ca_certs:
            uvicorn_options["ssl_ca_certs"] = ssl_ca_certs
        if ssl_ciphers:
            uvicorn_options["ssl_ciphers"] = ssl_ciphers

        if psutil.WINDOWS:
            uvicorn_options["loop"] = "asyncio"
            import asyncio

            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

        config = uvicorn.Config(svc.asgi_app, **uvicorn_options)
        uvicorn.Server(config).run(sockets=[sock])
    else:
        raise ValueError(f"Unsupported bind scheme: {bind}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
