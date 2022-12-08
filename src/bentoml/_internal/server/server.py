"""
Server class for getting the Bento client and managing server process
"""

from __future__ import annotations

import subprocess


class Server:
    def __init__(self, process: subprocess.Popen[bytes], host: str, port: int) -> None:
        self._process = process
        self._host = host
        self._port = port

    def get_client(self):
        from bentoml.client import Client

        Client.wait_until_server_is_ready(self._host, self._port, 10)
        return Client.from_url(f"http://localhost:{self._port}")

    def stop(self) -> None:
        self.process.kill()

    @property
    def process(self) -> subprocess.Popen[bytes]:
        return self._process

    @property
    def address(self) -> str:
        return f"{self._host}:{self._port}"
