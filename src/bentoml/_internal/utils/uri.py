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

# RFC 6598 Shared Address Space (CGNAT), e.g. used for carrier-grade NAT and internal
# cloud load balancers. Not classified as private/loopback/link-local/reserved by
# ipaddress, so it must be checked explicitly (GHSA-mrmq-3q62-6cc8 follow-up).
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


def is_unsafe_address(
    ip: "ipaddress.IPv4Address | ipaddress.IPv6Address",
) -> bool:
    """Whether `ip` must be blocked from outbound connections initiated by BentoML
    (e.g. when fetching a user-supplied URL). Covers private, loopback, link-local,
    and reserved ranges, plus the IPv4 CGNAT range which `ipaddress` does not
    classify as any of the above."""
    # Canonicalize IPv4-mapped IPv6 addresses (e.g. ::ffff:100.64.1.1) to their
    # embedded IPv4 form before checking. `is_private`/`is_loopback`/etc already
    # see through the mapping on CPython >= 3.10, but the CGNAT range below is a
    # manual membership check that doesn't, so a mapped CGNAT address would
    # otherwise slip past this guard.
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or (ip.version == 4 and ip in _CGNAT_NETWORK)
    )


@contextlib.contextmanager
def make_safe_connect():
    """Patch loop.create_connection() method to reject unsafe URLs."""

    from urllib.request import getproxies

    import httpx
    from uvloop import Loop

    from bentoml.exceptions import BadInput

    global original_create_connection

    if original_create_connection is None:
        original_create_connection = Loop.create_connection

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
                if is_unsafe_address(ip):
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
