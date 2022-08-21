from __future__ import annotations

import os
import sys
import json
import typing as t

import click


@click.command()
@click.argument("bento_identifier", type=click.STRING, required=False, default=".")
@click.option("--runner-name", type=click.STRING, required=True)
@click.option("--bind", type=click.STRING, required=True)
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
    required=False,
    type=click.INT,
    default=None,
    help="If set, start the server as a bare worker with the given worker ID. Otherwise start a standalone server with a supervisor process.",
)
@click.option(
    "--worker-env-map",
    required=False,
    type=click.STRING,
    default=None,
    help="The environment variables to pass to the worker process. The format is a JSON string, e.g. '{0: {\"CUDA_VISIBLE_DEVICES\": 0}}'.",
)
@click.pass_context
def main(
    ctx: click.Context,
    bento_identifier: str,
    runner_name: str,
    bind: str,
    working_dir: t.Optional[str],
    no_access_log: bool,
    worker_id: int | None,
    worker_env_map: str | None,
) -> None:
    """
    Start a runner server.

    Args:
        bento_identifier: the Bento identifier
        name: the name of the runner
        bind: the bind address URI. Can be:
            - tcp://host:port
            - unix://path/to/unix.sock
            - file:///path/to/unix.sock
            - fd://12
        working_dir: (Optional) the working directory
        worker_id: (Optional) if set, the runner will be started as a worker with the given ID. Important: begin from 1.
        worker_env_map: (Optional) the environment variables to pass to the worker process. The format is a JSON string, e.g. '{0: {\"CUDA_VISIBLE_DEVICES\": 0}}'.
    """
    if worker_id is None:

        # Start a standalone server with a supervisor process
        from circus.watcher import Watcher

        from bentoml_cli.utils import unparse_click_params
        from bentoml._internal.utils.circus import create_standalone_arbiter
        from bentoml._internal.utils.circus import create_circus_socket_from_uri

        circus_socket = create_circus_socket_from_uri(bind, name=runner_name)
        params = ctx.params
        params["bind"] = f"fd://$(circus.sockets.{runner_name})"
        params["worker_id"] = "$(circus.wid)"
        params["no_access_log"] = no_access_log
        watcher = Watcher(
            name=f"runner_{runner_name}",
            cmd=sys.executable,
            args=["-m", "bentoml_cli.server.runner"]
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
    from urllib.parse import urlparse

    import psutil

    from bentoml import load
    from bentoml._internal.log import configure_server_logging
    from bentoml._internal.context import component_context
    from bentoml._internal.utils.uri import uri_to_path

    configure_server_logging()
    import uvicorn  # type: ignore

    if no_access_log:
        from bentoml._internal.configuration.containers import BentoMLContainer

        access_log_config = BentoMLContainer.runners_config.logging.access
        access_log_config.enabled.set(False)

    from bentoml._internal.server.runner_app import RunnerAppFactory

    service = load(bento_identifier, working_dir=working_dir, standalone_load=True)

    # setup context
    component_context.component_type = "runner"
    component_context.component_name = runner_name
    component_context.component_index = worker_id
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

    app = RunnerAppFactory(runner, worker_index=worker_id)()

    parsed = urlparse(bind)
    uvicorn_options: dict[str, int | None | str] = {
        "log_config": None,
        "workers": 1,
    }

    if psutil.WINDOWS:
        uvicorn_options["loop"] = "asyncio"
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore

    if parsed.scheme in ("file", "unix"):
        uvicorn.run(
            app,  # type: ignore
            uds=uri_to_path(bind),
            **uvicorn_options,
        )
    elif parsed.scheme == "tcp":
        assert parsed.hostname is not None
        assert parsed.port is not None
        uvicorn.run(
            app,  # type: ignore
            host=parsed.hostname,
            port=parsed.port,
            **uvicorn_options,
        )
    elif parsed.scheme == "fd":
        # when fd is provided, we will skip the uvicorn internal supervisor, thus there is only one process
        fd = int(parsed.netloc)
        sock = socket.socket(fileno=fd)
        config = uvicorn.Config(app, **uvicorn_options)
        uvicorn.Server(config).run(sockets=[sock])
    else:
        raise ValueError(f"Unsupported bind scheme: {bind}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
