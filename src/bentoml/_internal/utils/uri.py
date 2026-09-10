import contextlib
import ipaddress
import os
import pathlib
import socket
from typing import no_type_check
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlparse
from urllib.request import url2pathname

import psutil


def path_to_uri(path: str) -> str:
    """
    Convert a path to a URI.

    Args:
        path: Path to convert to URI.

    Returns:
        URI string. (quoted, absolute)
    """
    path = os.path.abspath(path)
    if psutil.WINDOWS:
        return pathlib.PureWindowsPath(path).as_uri()
    if psutil.POSIX:
        return pathlib.PurePosixPath(path).as_uri()
    raise ValueError("Unsupported OS")


def uri_to_path(uri: str) -> str:
    """
    Convert a file URI to a path.

    Args:
        uri: URI to convert to path.

    Returns:
        Path string. (unquoted)
    """
    parsed = urlparse(uri)
    if parsed.scheme not in ("file", "filesystem", "unix"):
        raise ValueError("Unsupported URI scheme")
    host = "{0}{0}{mnt}{0}".format(os.path.sep, mnt=parsed.netloc)
    return os.path.normpath(os.path.join(host, url2pathname(unquote(parsed.path))))


def encode_path_for_uri(path: str) -> str:
    """Percent-encode non-URL characters in a path."""
    return quote(path.replace(os.sep, "/"))


def is_http_url(url: str) -> bool:
    return urlparse(url).scheme in {"http", "https"}


original_create_connection = None
_original_getaddrinfo = None


@contextlib.contextmanager
def make_safe_connect():
    """Patch network connection to reject requests to private/internal IP addresses.

    On Linux/macOS with uvloop: patches uvloop.Loop.create_connection.
    On Windows or without uvloop: patches socket.getaddrinfo as fallback.
    """

    from urllib.request import getproxies

    import httpx

    from bentoml.exceptions import BadInput

    try:
        from uvloop import Loop
    except ImportError:
        Loop = None

    global original_create_connection
    global _original_getaddrinfo

    # Do not check connections with proxy servers
    proxies = [
        (parsed.hostname, parsed.port)
        for parsed in map(urlparse, getproxies().values())
    ]

    if Loop is None:
        # Fallback for platforms without uvloop (e.g. Windows):
        # Patch socket.getaddrinfo to check resolved IPs before connection.
        if _original_getaddrinfo is None:
            _original_getaddrinfo = socket.getaddrinfo

        def safe_getaddrinfo(host, port, *args, **kwargs):
            results = _original_getaddrinfo(host, port, *args, **kwargs)
            if host is not None and (host, port) not in proxies:
                for family, type_, proto, canonname, sockaddr in results:
                    try:
                        ip = ipaddress.ip_address(sockaddr[0])
                    except ValueError:
                        continue
                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        raise socket.gaierror(
                            f"Blocked private IP address {sockaddr[0]}"
                        )
            return results

        socket.getaddrinfo = safe_getaddrinfo
        try:
            yield
        except httpx.ConnectError as e:
            if "All connection attempts failed" in str(e):
                raise BadInput("Connection blocked due to insecure input URL") from e
        finally:
            socket.getaddrinfo = _original_getaddrinfo
        return

    # uvloop available: use original Loop.create_connection patching
    if original_create_connection is None:
        original_create_connection = Loop.create_connection

    @no_type_check
    async def safe_create_connection(
        self, protocol_factory, host=None, port=None, **kwargs
    ):
        if host is not None and (host, port) not in proxies:
            try:
                ip = ipaddress.ip_address(host)
            except ValueError:
                raise socket.gaierror(f"Blocked invalid IP address {host}")
            else:
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    raise socket.gaierror(f"Blocked private IP address {host}")
        return await original_create_connection(
            self, protocol_factory, host=host, port=port, **kwargs
        )

    Loop.create_connection = safe_create_connection
    try:
        yield
    except httpx.ConnectError as e:
        if "All connection attempts failed" in str(e):
            raise BadInput("Connection blocked due to insecure input URL") from e
    finally:
        Loop.create_connection = original_create_connection


def join_paths(*paths: str) -> str:
    """Join multiple paths into a single path, ensuring proper separators."""
    result = ""
    for path in paths:
        if not path:
            continue
        result = result.rstrip("/") + "/" + path.lstrip("/")
    return result or "/"
