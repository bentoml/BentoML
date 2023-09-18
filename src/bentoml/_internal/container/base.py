from __future__ import annotations

import os
import sys
import typing as t
import logging
import subprocess
from abc import abstractmethod
from queue import Queue
from typing import TYPE_CHECKING
from itertools import chain
from threading import Thread

import attr

from ..utils import resolve_user_filepath
from ...exceptions import BentoMLException

if sys.version_info >= (3, 8):
    from functools import singledispatchmethod
else:
    from singledispatchmethod import singledispatchmethod


if TYPE_CHECKING:
    from typing_extensions import Self

    from ..types import PathType

    P = t.ParamSpec("P")

    ArgType: t.TypeAlias = tuple[str, ...] | None

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    ListStr = list[str]
else:
    ListStr = list


class Arguments(ListStr):
    def __add__(self: Self, other: Arguments) -> Arguments:
        return Arguments(super().__add__(other))

    @singledispatchmethod
    def construct_args(self, args: t.Any, opt: str = ""):
        raise NotImplementedError

    @construct_args.register(type(None))
    @construct_args.register(tuple)
    @construct_args.register(list)
    def _(self, args: ArgType, opt: str = ""):
        if args is not None:
            self.extend(
                list(chain.from_iterable(map(lambda arg: (f"--{opt}", arg), args)))
            )

    @construct_args.register(type(None))
    @construct_args.register(str)
    @construct_args.register(os.PathLike)
    def _(self, args: PathType, opt: str = ""):
        if args is not None:
            if os.path.exists(str(args)):
                args = resolve_user_filepath(str(args), ctx=None)
            self.extend((f"--{opt}", str(args)))

    @construct_args.register(type(None))
    @construct_args.register(bool)
    def _(self, args: bool, opt: str = ""):
        if args:
            self.append(f"--{opt}")


@attr.define(repr=False, init=False)
class OCIBuilder:
    """
    Base class for OCI builders. This is a command class
    that can be used to construct build commands for the given OCI builder.

    To implement a new OCI builder, provide a build and health check command via
    'container.register_backend()' instead.
    """

    binary: str = attr.field()
    env: dict[str, str] = attr.field(
        default=None, converter=attr.converters.default_if_none(factory=dict)
    )
    build_cmd: t.Sequence[str] = attr.field(
        default=None, converter=attr.converters.default_if_none(Arguments(["build"]))
    )
    enable_buildkit: bool = attr.field(default=True)

    def __repr__(self):
        return f"<OCIBuilder (binary={self.binary}, enable_buildkit={self.enable_buildkit})>"

    def __init__(
        self,
        binary: str,
        enable_buildkit: bool = True,
        build_cmd: t.Sequence[str] | None = None,
        env: dict[str, str] | None = None,
        *,
        _internal: bool = False,
    ):
        """
        Initialize the OCI builder.

        Args:
            backend: The name of the OCI builder.
            env: Environment variables to be passed to the OCI builder.
            enable_buildkit: Whether to enable BuildKit support for given OCI builder.
            build_cmd: The build command to be used by the OCI builder, minus the backend name.
                       Example: 'docker buildx build' -> ['buildx', 'build']
                       If not provided, the default will be ['build']
        """
        if not _internal:
            raise BentoMLException(
                "'OCIBuilder' cannot be instantiated directly, use 'container.get_backend()' instead."
            )
        self.__attrs_init__(binary, env, build_cmd, enable_buildkit)

    @staticmethod
    def create(
        binary: str,
        build_cmd: t.Sequence[str] | None = None,
        enable_buildkit: bool = True,
        env: dict[str, str] | None = None,
        *,
        health: t.Callable[[], bool],
        construct_build_args: t.Callable[P, list[str]],
    ) -> FrozenOCIBuilder:
        """
        Create an implementation of an OCI builder backend.
        """
        impl = FrozenOCIBuilder(
            env=env,
            binary=binary,
            build_cmd=build_cmd,
            enable_buildkit=enable_buildkit,
            _internal=True,
        )
        impl.construct_build_args = construct_build_args
        impl.health = health
        return impl

    @abstractmethod
    def health(self) -> bool:
        """
        Check if the given OCI builder is healthy.
        """
        raise NotImplementedError

    @abstractmethod
    def construct_build_args(self, **kwargs: t.Any) -> t.Sequence[str]:
        """
        Build the command for the given OCI builder.
        """
        raise NotImplementedError

    def build(self, **attrs: t.Any) -> bytes | None:
        """
        Build the OCI image, and returns the output.
        """
        context_path = attrs.get("context_path", None)
        use_sudo = attrs.pop("_run_as_root", False)
        assert context_path is not None, "context_path must be provided."
        cmds = Arguments([self.binary, *self.build_cmd])
        if use_sudo:
            cmds = ["sudo"] + cmds
        cmds.extend(self.construct_build_args(**attrs))
        logger.debug(
            "%s build cmd: '%s'", self.binary.split(os.sep)[-1], " ".join(cmds)
        )

        # We will always update suprocess environment into
        # default builder environment. This ensure that we
        # respect the user's environment.
        subprocess_env = os.environ.copy()
        env = self.env.copy()
        env.update(subprocess_env)
        commands = list(map(str, cmds))

        try:
            # We need to also respect DOCKER_BUILDKIT environment here
            # to stream logs
            if not self.enable_buildkit or (
                "DOCKER_BUILDKIT" in env and env["DOCKER_BUILDKIT"] == "0"
            ):
                return self.stream_logs(commands, cwd=context_path, env=env).stdout
            else:
                return subprocess.check_output(commands, cwd=context_path, env=env)
        except subprocess.CalledProcessError as e:
            if e.stderr:
                raise BentoMLException(e.stderr.decode("utf-8")) from None
            raise BentoMLException(str(e)) from None

    def stream_logs(
        self, cmds: t.Sequence[str], *, env: dict[str, str], cwd: PathType
    ) -> subprocess.CompletedProcess[bytes]:
        proc = subprocess.Popen(
            cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd
        )
        queue: Queue[tuple[str, bytes]] = Queue()
        stderr, stdout = b"", b""
        # We will use a thread to read from the subprocess and avoid hanging from Ctrl+C
        t = Thread(target=self._enqueue_output, args=(proc.stdout, "stdout", queue))
        t.daemon = True
        t.start()
        t = Thread(target=self._enqueue_output, args=(proc.stderr, "stderr", queue))
        t.daemon = True
        t.start()
        for _ in range(2):
            for src, line in iter(queue.get, None):
                logger.info(line.decode(errors="replace").strip("\n"))
                if src == "stderr":
                    stderr += line
                else:
                    stdout += line
        exit_code = proc.wait()
        if exit_code != 0:
            raise subprocess.CalledProcessError(
                exit_code, cmds, output=stdout, stderr=stderr
            )
        return subprocess.CompletedProcess(
            proc.args, exit_code, stdout=stdout, stderr=stderr
        )

    def _enqueue_output(
        self, pipe: t.IO[bytes], pipe_name: str, queue: Queue[tuple[str, bytes] | None]
    ):
        try:
            with pipe:
                for line in iter(pipe.readline, b""):
                    queue.put((pipe_name, line))
        finally:
            queue.put(None)


FrozenOCIBuilder = type(
    "FrozenOCIBuilder",
    (OCIBuilder,),
    {"__slots__": ("construct_build_args", "health")},
)
