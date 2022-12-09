from __future__ import annotations

import logging
import traceback
import subprocess
from typing import TYPE_CHECKING

import attr

from ..utils import cached_property

if TYPE_CHECKING:
    from types import TracebackType


logger = logging.getLogger(__name__)


@attr.frozen
class ServerHandle:
    process: subprocess.Popen[bytes]
    host: str
    port: int
    timeout: int = attr.field(default=10)

    @cached_property
    def client(self):
        return self.get_client()

    def get_client(self):
        from bentoml.client import Client

        Client.wait_until_server_is_ready(
            host=self.host, port=self.port, timeout=self.timeout
        )
        return Client.from_url(f"http://localhost:{self.port}")

    def stop(self) -> None:
        self.process.kill()

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
        traceback.print_exception(exc_type, exc_value, traceback_type)
