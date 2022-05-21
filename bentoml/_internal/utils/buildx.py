from __future__ import annotations

import os
import typing as t
import logging
import subprocess
from queue import Queue
from typing import TYPE_CHECKING
from functools import wraps
from threading import Thread

from ..types import PathType
from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    P = t.ParamSpec("P")
    from io import BytesIO


DOCKER_BUILDX_CMD = ["docker", "buildx"]


def postprocess_logs(func: t.Callable[P, t.Iterator[str]]):
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        iterator = func(*args, **kwargs)
        while True:
            try:
                print(next(iterator), end="")
            except subprocess.CalledProcessError as e:
                raise e
            except StopIteration:
                break

    return wrapper


def health() -> None:
    """
    Check whether buildx is available in given system.
    """
    cmds = DOCKER_BUILDX_CMD + ["--help"]
    try:
        output = subprocess.check_output(cmds)
        assert "buildx" in output.decode("utf-8")
    except (subprocess.CalledProcessError, AssertionError):
        raise BentoMLException(
            "BentoML requires Docker Buildx to be installed to support multi-arch builds. "
            "Buildx comes with Docker Desktop, but one can also install it manually by following "
            "instructions via https://docs.docker.com/buildx/working-with-buildx/#install."
        )


def list_builders() -> list[str]:
    cmds = DOCKER_BUILDX_CMD + ["ls"]
    proc = subprocess.run(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stream = proc.stdout.decode("utf-8")

    if len(stream) != 0 and stream[-1] == "\n":
        stream = stream[:-1]

    output = stream.splitlines()[1:]  # first line is a header
    # lines starting with a blank space are builders metadata, not builder name
    output = list(filter(lambda x: not x.startswith(" "), output))
    return [s.split(" ")[0] for s in output]


@postprocess_logs
def use(builder: str, default: bool = False, global_: bool = False) -> t.Iterator[str]:
    cmds = DOCKER_BUILDX_CMD + ["use"]
    if default:
        cmds.append("--default")
    if global_:
        cmds.append("--global")
    cmds.append(builder)
    return stream_buildx_logs(cmds)


@postprocess_logs
def create(
    subprocess_env: dict[str, str] | None = None,
    cwd: PathType | None = None,
    *,
    context_or_endpoints: str | None = None,
    buildkitd_flags: str | None = None,
    config: PathType | None = None,
    driver: t.Literal["docker", "kubernetes", "docker-container"] | None = None,
    driver_opt: dict[str, str] | None = None,
    name: str | None = None,
    platform: list[str] | None = None,
    use: bool = False,
) -> t.Iterator[str]:
    """
    Create a new buildx instance.

    Args:
        context_or_endpoints: Custom docker context or endpoints (DOCKER_HOSTS).
        buildkitd_flags: Flags to pass to buildkitd.
        config: Path to a buildx configuration file.
        driver: Driver to use for buildx.
        driver_opt: Driver options.
        name: Name of the buildx context.
        platform: List of platform for a given builder instance.
        use: whether to use the builder instance after create.
    """
    cmds = DOCKER_BUILDX_CMD + ["create"]

    if buildkitd_flags is not None:
        cmds.extend(["--buildkitd-flags", buildkitd_flags])
    if config is not None:
        cmds.extend(["--config", str(config)])
    if driver is not None:
        cmds.extend(["--driver", driver])
    if driver_opt is not None:
        cmds.extend(
            ["--driver-opt", ",".join([f"{k}={v}" for k, v in driver_opt.items()])]
        )

    if name is not None:
        cmds.extend(["--name", name])

    if platform is None:
        platform = [
            "linux/amd64",
            "linux/arm64/v8",
            "linux/ppc64le",
            "linux/s390x",
            "linux/riscv64",
            "linux/mips64le",
        ]
    cmds.extend(["--platform", ",".join(platform)])

    if use:
        cmds.append("--use")

    if context_or_endpoints is not None:
        cmds.append(context_or_endpoints)

    return stream_buildx_logs(cmds, env=subprocess_env, cwd=cwd)


@postprocess_logs
def build(
    subprocess_env: dict[str, str] | None,
    cwd: PathType | None,
    *,
    context_path: PathType = ".",
    add_host: dict[str, str] | None,
    allow: list[str] | None,
    build_args: dict[str, str] | None,
    build_context: dict[str, str] | None,
    builder: str | None,
    cache_from: str | dict[str, str] | None,
    cache_to: str | dict[str, str] | None,
    cgroup_parent: str | None,
    file: PathType | None,
    iidfile: PathType | None,
    labels: dict[str, str] | None,
    load: bool,
    metadata_file: PathType | None,
    network: str | None,
    no_cache: bool,
    no_cache_filter: list[str] | None,
    output: str | dict[str, str] | None,
    platform: str | list[str] | None,
    progress: t.Literal["auto", "tty", "plain"],
    pull: bool,
    push: bool,
    quiet: bool,
    secrets: str | list[str] | None,
    shm_size: int | None,
    rm: bool,
    ssh: str | None,
    tags: str | list[str] | None,
    target: str | None,
    ulimit: dict[str, str] | None,
) -> t.Iterator[str]:
    cmds = DOCKER_BUILDX_CMD + ["build"]

    cmds += ["--progress", progress]

    if tags is None:
        tags = []
    tags = [tags] if not isinstance(tags, list) else tags
    for tag in tags:
        cmds.extend(["--tag", tag])

    if add_host is not None:
        hosts = [f"{k}:{v}" for k, v in add_host.items()]
        for host in hosts:
            cmds.extend(["--add-host", host])

    if allow is not None:
        for allow_arg in allow:
            cmds.extend(["--allow", allow_arg])

    if build_args is not None:
        args = [f"{k}={v}" for k, v in build_args.items()]
        for arg in args:
            cmds.extend(["--build-arg", arg])

    if build_context is not None:
        args = [f"{k}={v}" for k, v in build_context.items()]
        for arg in args:
            cmds.extend(["--build-context", arg])

    if builder is not None:
        cmds.extend(["--builder", builder])

    if cache_from is not None:
        if isinstance(cache_from, str):
            cmds.extend(["--cache-from", cache_from])
        else:
            args = [f"{k}={v}" for k, v in cache_from.items()]
            cmds.extend(["--cache-from", ",".join(args)])

    if cache_to is not None:
        if isinstance(cache_to, str):
            cmds.extend(["--cache-to", cache_to])
        else:
            args = [f"{k}={v}" for k, v in cache_to.items()]
            cmds.extend(["--cache-to", ",".join(args)])

    if cgroup_parent is not None:
        cmds.extend(["--cgroup-parent", cgroup_parent])

    if file is not None:
        cmds.extend(["--file", str(file)])

    if iidfile is not None:
        cmds.extend(["--iidfile", str(iidfile)])

    if load:
        cmds.append("--load")

    if metadata_file is not None:
        cmds.extend(["--metadata-file", str(metadata_file)])

    if network is not None:
        cmds.extend(["--network", network])

    if no_cache:
        cmds.append("--no-cache")

    if no_cache_filter is not None:
        for arg in no_cache_filter:
            cmds.extend(["--no-cache-filter", arg])

    if labels is not None:
        args = [f"{k}={v}" for k, v in labels.items()]
        for arg in args:
            cmds.extend(["--label", arg])

    if output is not None:
        if isinstance(output, str):
            cmds.extend(["--output", output])
        else:
            args = [f"{k}={v}" for k, v in output.items()]
            cmds += ["--output", ",".join(args)]

    if platform is not None:
        if isinstance(platform, str):
            platform = [platform]
        cmds += ["--platform", ",".join(platform)]

    if pull:
        cmds.append("--pull")

    if push:
        cmds.append("--push")

    if quiet:
        cmds.append("--quiet")

    if secrets is not None:
        if isinstance(secrets, str):
            cmds.extend(["--secret", secrets])
        else:
            for secret in secrets:
                cmds.extend(["--secret", secret])

    if rm:
        cmds.append("--rm")

    if shm_size is not None:
        cmds.extend(["--shm-size", str(shm_size)])

    if ssh is not None:
        cmds.extend(["--ssh", ssh])

    if target is not None:
        cmds.extend(["--target", target])

    if ulimit is not None:
        if len(ulimit) > 1:
            raise ValueError("ulimit must be a dict with one key value for nofile.")
        args = [f"{k}={v}" for k, v in ulimit.items()]
        cmds.extend(["--ulimit", args[0]])

    cmds.append(str(context_path))

    return stream_buildx_logs(cmds, env=subprocess_env, cwd=cwd)


def stream_stdout_stderr(
    process: subprocess.Popen[t.Any],
) -> t.Iterable[tuple[str, bytes]]:
    q: Queue[t.Any] = Queue()
    stderr = b""  # error message

    def reader(pipe: BytesIO, pipe_name: str, q: Queue[t.Any]):
        try:
            with pipe:
                for line in iter(pipe.readline, b""):
                    q.put((pipe_name, line))
        finally:
            q.put(None)  # type: ignore

    # daemon threads to avoid hanging with Ctrl-C
    thread = Thread(target=reader, args=[process.stdout, "stdout", q], daemon=True)
    thread.start()
    thread = Thread(target=reader, args=[process.stderr, "stderr", q], daemon=True)
    thread.start()
    for _ in range(2):
        for s, l in iter(q.get, None):
            yield s, l
            if s == "stderr":
                stderr += l
    exit_code = process.wait()
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, process.args, stderr)


def stream_buildx_logs(
    cmds: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: PathType | None = None,
) -> t.Iterator[str]:
    subprocess_env = os.environ.copy()
    if env is not None:
        subprocess_env.update(env)

    full_cmd = list(map(str, cmds))
    process = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=subprocess_env,
        cwd=cwd,
    )

    for _, value in stream_stdout_stderr(process):
        yield value.decode("utf-8")
