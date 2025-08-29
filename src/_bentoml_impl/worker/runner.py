from __future__ import annotations

import inspect
import json
import logging
import signal
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import anyio
import click

if TYPE_CHECKING:
    from _bentoml_sdk import Service as _Service

    Service = _Service[Any]

logger = logging.getLogger("bentoml.worker.runner")


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--service-name", type=click.STRING, required=False, default="")
@click.option(
    "--worker-id",
    required=False,
    type=click.INT,
    default=None,
    help="If set, start the server as a bare worker with the given worker ID. Otherwise start a standalone server with a supervisor process.",
)
@click.option(
    "--development-mode",
    type=click.BOOL,
    help="Run the process in development mode",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option("--args", type=click.STRING, help="Bento arguments dict for the service")
def main(
    bento_identifier: str,
    service_name: str,
    worker_id: int | None,
    development_mode: bool,
    args: str | None,
):
    """Start a worker for running custom commands."""

    from _bentoml_impl.loader import load
    from bentoml._internal.container import BentoMLContainer
    from bentoml._internal.context import server_context
    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.utils.args import set_arguments

    if args:
        set_arguments(json.loads(args))

    configure_server_logging()

    if worker_id is not None:
        server_context.worker_index = worker_id
    # standalone_build=True means to not restore cwd and model store
    service = load(bento_identifier, standalone_load=True)

    if service_name and service_name != service.name:
        service = service.find_dependent_by_name(service_name)
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

    anyio.run(start_service, service)


async def start_service(service: Service) -> None:
    from _bentoml_sdk.service import set_current_service
    from bentoml._internal.utils import expand_envs

    instance = service.inner()

    if cmd_getter := getattr(instance, "__command__", None):
        if not callable(cmd_getter):
            raise TypeError(
                f"__command__ must be a callable that returns a list of strings, got {type(cmd_getter)}"
            )
        cmd = cast("list[str]", cmd_getter())
    else:
        cmd = service.cmd
    assert cmd is not None, "must have a command"
    cmd = [expand_envs(c) for c in cmd]
    logger.info("Running service with command: %s", " ".join(cmd))
    # Call on_startup hook with optional ctx or context parameter
    for name, member in vars(service.inner).items():
        if callable(member) and getattr(member, "__bentoml_startup_hook__", False):
            logger.info("Running startup hook: %s", name)
            result = getattr(instance, name)()  # call the bound method
            if inspect.isawaitable(result):
                await result
                logger.info("Completed async startup hook: %s", name)
            else:
                logger.info("Completed startup hook: %s", name)
    set_current_service(instance)
    process = None

    def signal_handler(sig, *_):
        if process is not None:
            process.send_signal(sig)

    original_handler = signal.signal(signal.SIGTERM, signal_handler)
    try:
        process = await anyio.open_process(cmd, stdout=None, stderr=None)
        retcode = await process.wait()
        logger.info("Process exited with code %d", retcode)
    finally:
        signal.signal(signal.SIGTERM, original_handler)
        # Call on_shutdown hook with optional ctx or context parameter
        for name, member in vars(service.inner).items():
            if callable(member) and getattr(member, "__bentoml_shutdown_hook__", False):
                logger.info("Running cleanup hook: %s", name)
                result = getattr(instance, name)()  # call the bound method
                if inspect.isawaitable(result):
                    await result
                    logger.info("Completed async cleanup hook: %s", name)
                else:
                    logger.info("Completed cleanup hook: %s", name)
        set_current_service(None)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
