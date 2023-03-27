from __future__ import annotations

import sys
import typing as t
import logging
import subprocess
from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from bentoml.exceptions import StateException

from ._internal.tag import Tag
from ._internal.bento import Bento
from ._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from types import TracebackType

    from ._internal.client import Client
    from ._internal.client import GrpcClient
    from ._internal.client import HTTPClient


logger = logging.getLogger(__name__)


def is_running(
    process: subprocess.Popen[bytes] | None,
) -> t.TypeGuard[subprocess.Popen[bytes]]:
    return process is not None and process.poll() is None


class Server(ABC):
    bento: str | Bento | Tag
    host: str
    port: int

    args: list[str]

    process: subprocess.Popen[bytes] | None = None
    timeout: float = 10
    _client: Client | None = None

    def __init__(
        self,
        bento: str | Bento | Tag,
        serve_cmd: str,
        reload: bool,
        production: bool,
        env: t.Literal["conda"] | None,
        host: str,
        port: int,
        working_dir: str | None,
        api_workers: int | None,
        backlog: int,
    ):
        self.bento = bento

        if isinstance(bento, Bento):
            bento_str = str(bento.tag)
        elif isinstance(bento, Tag):
            bento_str = str(bento)
        else:
            bento_str = bento

        args: list[str] = [
            sys.executable,
            "-m",
            "bentoml",
            "serve-http",
            bento_str,
            "--host",
            host,
            "--port",
            str(port),
            "--backlog",
            str(backlog),
        ]

        if production:
            args.append("--production")
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
        self.start = self._create_startmanager()

    def _create_startmanager(server):  # type: ignore # not calling self self
        class _StartManager:
            def __init__(self, blocking: bool = False):
                logger.info(f"starting server with arguments: {server.args}")

                server.process = subprocess.Popen(
                    server.args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                )

                if blocking:
                    server.process.wait()

            def __enter__(self):
                return server.get_client()

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_value: BaseException | None,
                traceback: TracebackType | None,
            ):
                server.stop()

        return _StartManager

    @abstractmethod
    def get_client(self) -> Client:
        pass

    def stop(self) -> None:
        if not is_running(self.process):
            logger.warning("Attempted to stop a BentoML server that was not running!")
            return
        self.process.terminate()

    def __enter__(self):
        logger.warning(
            "Using bentoml.Server as a context manager is deprecated, use bentoml.Server.start instead."
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


class HTTPServer(Server):
    _client: HTTPClient | None = None

    @inject
    def __init__(
        self,
        bento: str | Bento | Tag,
        reload: bool = False,
        production: bool = False,
        env: t.Literal["conda"] | None = None,
        host: str = Provide[BentoMLContainer.http.host],
        port: int = Provide[BentoMLContainer.http.port],
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

        if api_workers is not None:
            self.args.extend(["--api-workers", str(api_workers)])
        if working_dir is not None:
            self.args.extend(["--working-dir", str(working_dir)])

    def client(self) -> HTTPClient:
        logger.warning(
            "'Server.client()' is deprecated, use 'Server.get_client()' instead."
        )
        return self.get_client()

    def get_client(self) -> HTTPClient:
        if not is_running(self.process):
            raise StateException(
                "Attempted to get a client for a server that isn't running! Try running `Server.start()` first."
            )

        if self._client is None:
            from .client import HTTPClient

            HTTPClient.wait_until_server_ready(
                host=self.host, port=self.port, timeout=self.timeout
            )
            self._client = HTTPClient.from_url(f"http://{self.host}:{self.port}")
        return self._client


class GrpcServer(Server):
    _client: GrpcClient | None = None

    @inject
    def __init__(
        self,
        bento: str | Bento | Tag,
        reload: bool = False,
        production: bool = False,
        env: t.Literal["conda"] | None = None,
        host: str = Provide[BentoMLContainer.http.host],
        port: int = Provide[BentoMLContainer.http.port],
        working_dir: str | None = None,
        api_workers: int | None = Provide[BentoMLContainer.api_server_workers],
        backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
        enable_reflection: bool = Provide[BentoMLContainer.grpc.reflection.enabled],
        enable_channelz: bool = Provide[BentoMLContainer.grpc.channelz.enabled],
        max_concurrent_streams: int
        | None = Provide[BentoMLContainer.grpc.max_concurrent_streams],
        grpc_protocol_version: str | None = None,
    ):
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
        )

        if enable_reflection:
            self.args.append("--enable-reflection")
        if enable_channelz:
            self.args.append("--enable-channelz")
        if max_concurrent_streams is not None:
            self.args.extend(["--max-concurrent-streams", str(max_concurrent_streams)])

        if grpc_protocol_version is not None:
            self.args.extend(["--protocol-version", str(grpc_protocol_version)])

    def get_client(self) -> GrpcClient:
        if not is_running(self.process):
            raise StateException(
                "Attempted to get a client for a server that isn't running! Try running `Server.start()` first."
            )

        if self._client is None:
            from .client import GrpcClient

            GrpcClient.wait_until_server_ready(
                host=self.host, port=self.port, timeout=self.timeout
            )
            self._client = GrpcClient.from_url(f"{self.host}:{self.port}")
        return self._client
