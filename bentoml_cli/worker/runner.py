from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asgiref.typing import ASGI3Application

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option(
    "--runner-name",
    type=click.STRING,
    required=True,
    envvar="RUNNER_NAME",
)
@click.option("--fd", type=click.INT, required=True)
@click.option("--working-dir", required=False, default=None, help="Working directory")
@click.option(
    "--no-access-log",
    required=False,
    type=click.BOOL,
    is_flag=True,
    default=False,
    help="Disable the runner server's access log",
)
@click.option(
    "--worker-id",
    required=True,
    type=click.INT,
    default=None,
    help="If set, start the server as a bare worker with the given worker ID. Otherwise start a standalone server with a supervisor process.",
)
def main(
    bento_identifier: str,
    runner_name: str,
    fd: int,
    working_dir: t.Optional[str],
    no_access_log: bool,
    worker_id: int,
) -> None:
    """
    Start a runner worker.
    """

    import socket

    import psutil

    from bentoml import load
    from bentoml._internal.context import component_context

    component_context.component_name = f"runner:{runner_name}:{worker_id}"
    from bentoml._internal.log import configure_server_logging

    configure_server_logging()
    import uvicorn  # type: ignore

    if no_access_log:
        from bentoml._internal.configuration.containers import BentoMLContainer

        access_log_config = BentoMLContainer.runners_config.logging.access
        access_log_config.enabled.set(False)

    from bentoml._internal.server.runner_app import RunnerAppFactory

    service = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    # setup context
    if service.tag is None:
        component_context.bento_name = f"*{service.__class__}"
        component_context.bento_version = "not available"
    else:
        component_context.bento_name = service.tag.name
        component_context.bento_version = service.tag.version

    for runner in service.runners:
        if runner.name == runner_name:
            break
    else:
        raise ValueError(f"Runner {runner_name} not found")

    app = t.cast("ASGI3Application", RunnerAppFactory(runner, worker_index=worker_id)())

    uvicorn_options: dict[str, int | None | str] = {
        "log_config": None,
        "workers": 1,
    }

    if psutil.WINDOWS:
        uvicorn_options["loop"] = "asyncio"
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

    sock = socket.socket(fileno=fd)
    config = uvicorn.Config(app, **uvicorn_options)
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
