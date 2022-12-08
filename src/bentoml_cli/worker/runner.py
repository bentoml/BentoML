from __future__ import annotations

import os
import sys
import json
import typing as t

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
    help="If set, start the server as a bare worker with the given worker ID. Otherwise start a standalone server with a supervisor process.",
)
@click.option(
    "--worker-env-map",
    required=False,
    type=click.STRING,
    default=None,
    help="The environment variables to pass to the worker process. The format is a JSON string, e.g. '{0: {\"CUDA_VISIBLE_DEVICES\": 0}}'.",
)
@click.option(
    "--prometheus-dir",
    type=click.Path(exists=True),
    help="Required by prometheus to pass the metrics in multi-process mode",
)
def main(
    bento_identifier: str,
    runner_name: str,
    fd: int,
    working_dir: t.Optional[str],
    no_access_log: bool,
    worker_id: int,
    worker_env_map: str | None,
    prometheus_dir: str | None,
) -> None:
    """
    Start a runner server.

    Args:
        bento_identifier: the Bento identifier
        name: the name of the runner
        fd: the file descriptor of the runner server's socket
        working_dir: (Optional) the working directory
        worker_id: (Optional) if set, the runner will be started as a worker with the given ID. Important: begin from 1.
        worker_env_map: (Optional) the environment variables to pass to the worker process. The format is a JSON string, e.g. '{0: {\"CUDA_VISIBLE_DEVICES\": 0}}'.
    """

    # setting up the environment for the worker process
    assert (
        "bentoml" not in sys.modules
    ), "bentoml should not be imported before setting up the environment, otherwise some of the environment may not take effect"
    if worker_env_map is not None:
        env_map: dict[str, dict[str, t.Any]] = json.loads(worker_env_map)
        worker_key = str(worker_id - 1)  # the worker ID is 1-based
        assert (
            worker_key in env_map
        ), f"worker_id {repr(worker_key)} not found in worker_env_map: {worker_env_map}"
        os.environ.update(env_map[worker_key])

    import socket

    import psutil

    from bentoml import load
    from bentoml._internal.context import component_context

    # setup context
    component_context.component_type = "runner"
    component_context.component_name = runner_name
    component_context.component_index = worker_id

    from bentoml._internal.log import configure_server_logging

    configure_server_logging()

    import uvicorn

    from bentoml._internal.configuration.containers import BentoMLContainer

    if prometheus_dir is not None:
        BentoMLContainer.prometheus_multiproc_dir.set(prometheus_dir)

    if no_access_log:
        access_log_config = BentoMLContainer.runners_config.logging.access
        access_log_config.enabled.set(False)

    from bentoml._internal.server.runner_app import RunnerAppFactory

    service = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    # setup context
    if service.tag is None:
        component_context.bento_name = service.name
        component_context.bento_version = "not available"
    else:
        component_context.bento_name = service.tag.name
        component_context.bento_version = service.tag.version or "not available"

    for runner in service.runners:
        if runner.name == runner_name:
            break
    else:
        raise ValueError(f"Runner {runner_name} not found")

    app = RunnerAppFactory(runner, worker_index=worker_id)()

    uvicorn_options: dict[str, int | None | str] = {
        "log_config": None,
        "workers": 1,
        "timeout_keep_alive": 1800,
    }

    if psutil.WINDOWS:
        # 1. uvloop is not supported on Windows
        # 2. the default policy for Python > 3.8 on Windows is ProactorEventLoop, which doesn't
        #    support listen on a existing socket file descriptors
        # See https://docs.python.org/3.8/library/asyncio-platforms.html#windows
        uvicorn_options["loop"] = "asyncio"
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

    # when fd is provided, we will skip the uvicorn internal supervisor, thus there is only one process
    sock = socket.socket(fileno=fd)
    config = uvicorn.Config(app, **uvicorn_options)
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
