from __future__ import annotations

import signal
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from typing import Optional
from urllib.parse import parse_qs


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")


def _method_with_timeout(your_method, timeout_seconds=5, *args, **kwargs):
    if timeout_seconds is not None:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        result = your_method(*args, **kwargs)
    except TimeoutException as te:
        raise te
    finally:
        if timeout_seconds is not None:
            signal.alarm(0)  # Reset the alarm

    return result


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

    # # noinspection PyPep8Naming
    # def do_GET(self) -> None:
    #     """
    #     Provides callback page for the oauth2 redirect
    #     """
    #     try:
    #         params = parse_qs(urlparse(self.path).query)

    #         has_error = ("code" not in params
    #                     or len(params['code']) != 1
    #                     or params['code'][0].strip() == "")

    #         if has_error:
    #             self.send_response(400, "Something went wrong trying to authenticate you. Please try going back in your browser, or restart the auth process.")
    #             self.end_headers()
    #         else:
    #             self.send_response(200, "You have been authenticated successfully. You may close this browser window now and go back to the terminal.")
    #             self.end_headers()
    #             self.server._code = params["code"][0]
    #     except:
    #         self.send_response(500)
    #         self.end_headers()

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

    def __init__(self, port):
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

    def wait_for_code(self, attempts: int = 3, timeout_per_attempt=10) -> Optional[str]:
        """
        Wait for the server to callback from token provider.

        It tries for #attempts with a timeout of #timeout_per_attempts for each attempt.
        This prevents the CLI from getting stuck by unsolved callback URls

        :param attempts: Amount of attempts
        :param timeout_per_attempt: Timeout for each attempt to be successful
        :return: Code from callback page or None if the callback page is not called successfully
        """
        for _ in range(0, attempts):
            try:
                _method_with_timeout(
                    self.handle_request, timeout_seconds=timeout_per_attempt
                )
            except TimeoutException:
                continue
            if self.get_code() is not None:
                return self.get_code()

        return None

    def wait_indefinitely_for_code(self) -> Optional[str]:
        """
        Wait indefinitely for ther server to callback from token provider.
        """
        self.handle_request()
        return self.get_code()
