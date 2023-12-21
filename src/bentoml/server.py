from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import textwrap
import typing as t
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING
from warnings import warn

from simple_di import Provide
from simple_di import inject

from ._internal.bento import Bento
from ._internal.client import Client
from ._internal.client.grpc import GrpcClient
from ._internal.client.http import HTTPClient
from ._internal.configuration.containers import BentoMLContainer
from ._internal.service import Service
from ._internal.tag import Tag
from ._internal.utils.analytics.usage_stats import BENTOML_SERVE_FROM_SERVER_API
from .exceptions import InvalidArgument
from .exceptions import ServerStateException
from .exceptions import UnservableException

if TYPE_CHECKING:
    from types import TracebackType

    from _bentoml_sdk import Service as NewService

    _FILE: t.TypeAlias = None | int | t.IO[t.Any]


STOP_TIMEOUT = 5
logger = logging.getLogger(__name__)

__all__ = ["Server", "GrpcServer", "HTTPServer"]


ClientType = t.TypeVar("ClientType", bound=Client)


class Server(ABC, t.Generic[ClientType]):
    servable: str | Bento | Tag | Service | NewService[t.Any]
    host: str
    port: int

    args: list[str]

    process: subprocess.Popen[bytes] | None = None
    _client: Client | None = None

    def __init__(
        self,
        servable: str | Bento | Tag | Service | NewService[t.Any],
        serve_cmd: str,
        reload: bool,
        production: bool,
        env: t.Literal["conda"] | None,
        host: str,
        port: int,
        working_dir: str | None,
        api_workers: int | None,
        backlog: int,
        bento: str | Bento | Tag | Service | None = None,
        timeout: float = 10,
    ):
        if bento is not None:
            if servable is None:  # type: ignore  # dealing with backwards compatibility, where a user has set bento argument manually.
                warn(
                    f"serving using the 'bento' argument is deprecated, either remove it as a kwarg or pass '{bento}' as the first positional argument",
                    DeprecationWarning,
                    stacklevel=2,
                )
                servable = bento
            else:
                raise InvalidArgument(
                    "Cannot use both 'bento' and 'servable' arguments; as 'bento' is deprecated, set 'servable' instead."
                )

        self.servable = servable
        # backward compatibility
        self.bento = servable

        if isinstance(servable, Bento):
            bento_str = str(servable.tag)
        elif isinstance(servable, Tag):
            bento_str = str(servable)
        elif isinstance(servable, Service):
            if not servable.is_service_importable():
                raise UnservableException(
                    "Cannot use 'bentoml.Service' as a server if it is defined in interactive session or Jupyter Notebooks."
                )
            bento_str, working_dir = servable.get_service_import_origin()
        elif not isinstance(servable, str):
            bento_str = servable.import_string
            working_dir = servable.working_dir
        else:
            bento_str = servable

        args: list[str] = [
            sys.executable,
            "-m",
            "bentoml",
            serve_cmd,
            bento_str,
            "--host",
            host,
            "--port",
            str(port),
            "--backlog",
            str(backlog),
        ]

        if not production:
            args.append("--development")
        if reload:
            args.append("--reload")
        if env:
            args.extend(["--env", env])

        if api_workers is not None:
            args.extend(["--api-workers", str(api_workers)])
        if working_dir is not None:
            args.extend(["--working-dir", str(working_dir)])

        self.args = args
        self.host = "127.0.0.1" if host == "0.0.0.0" else host
        self.port = port
        self.timeout = timeout

    def start(
        self,
        blocking: bool = False,
        env: dict[str, str] | None = None,
        stdin: _FILE = None,
        stdout: _FILE = None,
        stderr: _FILE = None,
        text: bool | None = None,
    ) -> t.ContextManager[ClientType]:
        """Start the server programmatically.

        To get the client, use the context manager.

        .. note::
           ``blocking=True`` and using ``start()`` as a context manager is mutually exclusive.

        Args:
            blocking: If True, the server will block until it is stopped.
            env: A dictionary of environment variables to pass to the server. Default to ``None``.
            stdin: The stdin to pass to the server. Default to ``None``.
            stdout: The stdout to pass to the server. Default to ``None``.
            stderr: The stderr to pass to the server. Default to ``None``.
            text: Whether to output as text or bytes for stdout and stderr. Default to ``None``.
        """
        # NOTE: User shouldn't manually set this since this envvar will be managed by BentoML.
        os.environ[BENTOML_SERVE_FROM_SERVER_API] = str(True)

        if env is None:
            env = {}
        env.update(os.environ.copy())

        class _Manager:
            def __init__(__inner_self):
                logger.debug(f"Starting server with arguments: {self.args}")
                default_io_descriptor = None if blocking else subprocess.PIPE
                if text is None:
                    warn(
                        "Setting text to True will be the default behavior for bentoml 2.x. Set it explicitly to avoid breaking changes.\n"
                        + "For Example: 'server.start(text=False, ...)'"
                    )
                self.process = subprocess.Popen(
                    self.args,
                    stdout=stdout if stdout is not None else default_io_descriptor,
                    stderr=stderr if stderr is not None else default_io_descriptor,
                    stdin=stdin if stdin is not None else default_io_descriptor,
                    env=env,
                    # TODO: Make this default to True in 2.x
                    text=text if text is not None else False,
                )

                if blocking:
                    try:
                        self.process.wait()
                    except KeyboardInterrupt:
                        pass

            def __enter__(__inner_self) -> ClientType:
                return self.get_client()

            def __exit__(
                __inner_self,
                _exc_type: type[BaseException] | None,
                _exc_value: BaseException | None,
                _traceback: TracebackType | None,
            ):
                self.stop()

        return _Manager()

    def get_client(self) -> ClientType:
        if self.process is None:
            # NOTE: if the process is None, we reset this envvar
            del os.environ[BENTOML_SERVE_FROM_SERVER_API]
            raise ServerStateException(
                "Attempted to get a client for a BentoML server that was not running! Try running 'bentoml.*Server.start()' first."
            )
        assert self.process is not None
        out_code = self.process.poll()
        if out_code == 0:
            # NOTE: if the process is None, we reset this envvar
            del os.environ[BENTOML_SERVE_FROM_SERVER_API]
            raise ServerStateException(
                "Attempted to get a client from a BentoML server that has already exited! You can run '.start()' again to restart it."
            )
        elif out_code is not None:
            # NOTE: if the process is None, we reset this envvar
            del os.environ[BENTOML_SERVE_FROM_SERVER_API]
            logs = "Attempted to get a client from a BentoML server that has already exited with an error!\nServer Output:\n"
            if self.process.stdout is not None and not self.process.stdout.closed:
                s = self.process.stdout.read()
                logs += textwrap.indent(
                    s.decode("utf-8") if isinstance(s, bytes) else s,
                    " " * 4,  # type: ignore  # may be string
                )
            if self.process.stderr is not None and not self.process.stderr.closed:
                logs += "\nServer Error:\n"
                s = self.process.stderr.read()
                logs += textwrap.indent(
                    s.decode("utf-8") if isinstance(s, bytes) else s,
                    " " * 4,  # type: ignore  # may be string
                )
            raise ServerStateException(logs)
        return self._get_client()

    @abstractmethod
    def _get_client(self) -> ClientType:
        pass

    def stop(self) -> None:
        # NOTE: User shouldn't manually set this since this envvar will be managed by BentoML.
        del os.environ[BENTOML_SERVE_FROM_SERVER_API]

        if self.process is None:
            logger.warning("Attempted to stop a BentoML server that was not running!")
            return
        assert self.process is not None
        out_code = self.process.poll()
        if out_code == 0:
            logger.warning(
                "Attempted to stop a BentoML server that has already exited!"
            )
            return
        elif out_code is not None:
            logs = "Attempted to stop a BentoML server that has already exited with an error!\nServer Output:\n"
            if self.process.stdout is not None and not self.process.stdout.closed:
                s = self.process.stdout.read()
                logs += textwrap.indent(
                    s.decode("utf-8") if isinstance(s, bytes) else s,
                    " " * 4,  # type: ignore  # may be string
                )
            if self.process.stderr is not None and not self.process.stderr.closed:
                logs += "\nServer Error:\n"
                s = self.process.stderr.read()
                logs += textwrap.indent(
                    s.decode("utf-8") if isinstance(s, bytes) else s,
                    " " * 4,  # type: ignore  # may be string
                )
            logger.warning(logs)
            return

        if sys.platform == "win32":
            os.kill(self.process.pid, signal.CTRL_C_EVENT)
        else:
            self.process.terminate()
        try:
            # NOTE: To avoid zombie processes
            self.process.communicate(timeout=STOP_TIMEOUT)
        except KeyboardInterrupt:
            pass
        except subprocess.TimeoutExpired:
            self.process.kill()  # force kill
            self.process.wait()

    def __enter__(self):
        warn(
            "Using bentoml.Server as a context manager is deprecated, use bentoml.Server.start instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback_type: TracebackType,
    ):
        try:
            self.stop()
        except Exception as e:  # pylint: disable=broad-except
            logger.error(f"Error stopping server: {e}", exc_info=e)


class HTTPServer(Server[HTTPClient]):
    _client: HTTPClient | None = None

    @inject
    def __init__(
        self,
        bento: str | Bento | Tag | Service,
        reload: bool = False,
        production: bool = True,
        env: t.Literal["conda"] | None = None,
        host: str = Provide[BentoMLContainer.http.host],
        port: int = Provide[BentoMLContainer.http.port],
        timeout: float = 10,
        working_dir: str | None = None,
        api_workers: int | None = Provide[BentoMLContainer.api_server_workers],
        backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
        ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
        ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
        ssl_keyfile_password: str
        | None = Provide[BentoMLContainer.ssl.keyfile_password],
        ssl_version: int | None = Provide[BentoMLContainer.ssl.version],
        ssl_cert_reqs: int | None = Provide[BentoMLContainer.ssl.cert_reqs],
        ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
        ssl_ciphers: str | None = Provide[BentoMLContainer.ssl.ciphers],
    ):
        # hacky workaround to prevent bentoml.serve being overwritten immediately
        from .serve import construct_ssl_args

        super().__init__(
            bento,
            "serve-http",
            reload,
            production,
            env,
            host,
            port,
            working_dir,
            api_workers,
            backlog,
            timeout=timeout,
        )

        ssl_args: dict[str, t.Any] = {
            "ssl_certfile": ssl_certfile,
            "ssl_keyfile": ssl_keyfile,
            "ssl_ca_certs": ssl_ca_certs,
        }
        ssl_args.update(
            ssl_keyfile_password=ssl_keyfile_password,
            ssl_version=ssl_version,
            ssl_cert_reqs=ssl_cert_reqs,
            ssl_ciphers=ssl_ciphers,
        )

        self.args.extend(construct_ssl_args(**ssl_args))

    def get_client(self) -> HTTPClient:
        return super().get_client()

    def client(self) -> HTTPClient | None:
        warn(
            "'Server.client()' is deprecated, use 'Server.get_client()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get_client()

    def _get_client(self) -> HTTPClient:
        if self._client is None:
            from .client import HTTPClient

            HTTPClient.wait_until_server_ready(
                host=self.host, port=self.port, timeout=self.timeout
            )
            self._client = HTTPClient.from_url(f"http://{self.host}:{self.port}")
        return self._client


class GrpcServer(Server[GrpcClient]):
    _client: GrpcClient | None = None

    @inject
    def __init__(
        self,
        bento: str | Bento | Tag | Service,
        reload: bool = False,
        production: bool = True,
        env: t.Literal["conda"] | None = None,
        host: str = Provide[BentoMLContainer.grpc.host],
        port: int = Provide[BentoMLContainer.grpc.port],
        timeout: float = 10,
        working_dir: str | None = None,
        api_workers: int | None = Provide[BentoMLContainer.api_server_workers],
        backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
        enable_reflection: bool = Provide[BentoMLContainer.grpc.reflection.enabled],
        enable_channelz: bool = Provide[BentoMLContainer.grpc.channelz.enabled],
        max_concurrent_streams: int
        | None = Provide[BentoMLContainer.grpc.max_concurrent_streams],
        ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
        ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
        ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
        grpc_protocol_version: str | None = None,
    ):
        from .serve import construct_ssl_args

        super().__init__(
            bento,
            "serve-grpc",
            reload,
            production,
            env,
            host,
            port,
            working_dir,
            api_workers,
            backlog,
            timeout=timeout,
        )

        ssl_args: dict[str, t.Any] = {
            "ssl_certfile": ssl_certfile,
            "ssl_keyfile": ssl_keyfile,
            "ssl_ca_certs": ssl_ca_certs,
        }

        self.args.extend(construct_ssl_args(**ssl_args))

        if enable_reflection:
            self.args.append("--enable-reflection")
        if enable_channelz:
            self.args.append("--enable-channelz")
        if max_concurrent_streams is not None:
            self.args.extend(["--max-concurrent-streams", str(max_concurrent_streams)])

        if grpc_protocol_version is not None:
            self.args.extend(["--protocol-version", str(grpc_protocol_version)])

    def _get_client(self) -> GrpcClient:
        if self._client is None:
            from .client import GrpcClient

            GrpcClient.wait_until_server_ready(
                host=self.host, port=self.port, timeout=self.timeout
            )
            self._client = GrpcClient.from_url(f"{self.host}:{self.port}")
        return self._client
