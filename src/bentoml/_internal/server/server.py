from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

import attr

if TYPE_CHECKING:
    from types import TracebackType


logger = logging.getLogger(__name__)


@attr.frozen
class ServerHandle:
    process: subprocess.Popen[bytes]
    host: str
    port: int
    timeout: int = attr.field(default=10)

    def client(self):
        return self.get_client()

    def get_client(self):
        from ..client import Client

        client = Client.from_url(f"http://{self.host}:{self.port}")
        client.wait_until_server_ready(timeout=10)
        return client

    def stop(self) -> None:
        self.process.terminate()

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def __enter__(self):
        yield self

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
