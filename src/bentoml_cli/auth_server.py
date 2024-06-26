from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from typing import Optional
from urllib.parse import parse_qs


class AuthRedirectHandler(BaseHTTPRequestHandler):
    """
    HTTPRequest Handler that is intended to be used as oauth2 callback page
    """

    def log_message(self, format, *args):
        """
        Disables logging for BaseHTTPRequestHandler
        """
        # silence the log messages
        pass

    def end_headers(self):
        """Add necessary headers for CORS"""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        """Handle preflight CORS requests"""
        self.send_response(200)
        self.end_headers()

    def do_POST(self) -> None:
        """
        Provides callback page for the oauth2 redirect using POST request
        """
        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length).decode("utf-8")
            params = parse_qs(post_data)

            has_error = (
                "code" not in params
                or len(params["code"]) != 1
                or params["code"][0].strip() == ""
            )

            if has_error:
                self.send_response(400)
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.server._code = params["code"][0]
        except Exception:
            self.send_response(500)
            self.end_headers()


class AuthCallbackHttpServer(HTTPServer):
    """
    Simplistic HTTP Server to provide local callback URL for token provider
    """

    def __init__(self, port: int):
        super().__init__(("", port), AuthRedirectHandler)

        self._code: str | None = None

    def get_code(self) -> str | None:
        """
        This method should only be called after the request was done and might be None when no token is given.

        :return: Authorization code or None if the request was not performed yet
        """
        return self._code

    @property
    def callback_url(self) -> str:
        """
        Callback URL for the HTTP-Server
        """
        return f"http://localhost:{self.server_port}"

    def wait_indefinitely_for_code(self) -> Optional[str]:
        """
        Wait indefinitely for ther server to callback from token provider.
        """
        while self._code is None:
            self.handle_request()
        return self.get_code()
