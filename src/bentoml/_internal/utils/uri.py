import asyncio
import contextlib
import ipaddress
import os
import pathlib
import socket
import threading
from typing import Any
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


# Reference-counted state for the create_connection() guard, keyed by event
# loop class. Each entry maps the patched loop class to a tuple of
# (number of open guarded windows, saved original unbound create_connection).
_safe_connect_state: dict[type, tuple[int, Any]] = {}
# Event loop classes are process-global, and guarded windows may be entered
# from multiple threads (each running its own loop); this lock serializes
# install/restore transitions.
_safe_connect_lock = threading.Lock()


@contextlib.contextmanager
def make_safe_connect():
    """Patch loop.create_connection() method to reject unsafe URLs.

    The patch is reference-counted per event loop class: the first guarded
    window installs it, and overlapping windows (e.g. concurrent requests)
    only increment the reference count. An exiting window therefore never
    restores the original method while another window still relies on the
    guard; the restore happens only when the last window closes.
    """

    from urllib.request import getproxies

    import httpx

    from bentoml.exceptions import BadInput

    # Patch the class of the loop that is currently running, so that the
    # guard also applies to plain asyncio (or Windows) event loops instead
    # of only uvloop.
    loop_class = type(asyncio.get_running_loop())

    # Do not check connections with proxy servers
    proxies = [
        (parsed.hostname, parsed.port)
        for parsed in map(urlparse, getproxies().values())
    ]

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
        original = _safe_connect_state[loop_class][1]
        return await original(self, protocol_factory, host=host, port=port, **kwargs)

    with _safe_connect_lock:
        state = _safe_connect_state.get(loop_class)
        if state is None:
            # No guarded window is open for this loop class: install the
            # patch and remember the original method for the final restore.
            original = loop_class.create_connection
            _safe_connect_state[loop_class] = (1, original)
            loop_class.create_connection = safe_create_connection
        else:
            depth, original = state
            _safe_connect_state[loop_class] = (depth + 1, original)
    try:
        yield
    except httpx.ConnectError as e:
        if "All connection attempts failed" in str(e):
            raise BadInput("Connection blocked due to insecure input URL") from e
    finally:
        with _safe_connect_lock:
            depth, original = _safe_connect_state[loop_class]
            if depth <= 1:
                # This is the last open guarded window for this loop class:
                # it is now safe to restore the original method.
                del _safe_connect_state[loop_class]
                loop_class.create_connection = original
            else:
                _safe_connect_state[loop_class] = (depth - 1, original)


def join_paths(*paths: str) -> str:
    """Join multiple paths into a single path, ensuring proper separators."""
    result = ""
    for path in paths:
        if not path:
            continue
        result = result.rstrip("/") + "/" + path.lstrip("/")
    return result or "/"
