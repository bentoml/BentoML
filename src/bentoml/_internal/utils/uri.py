import ipaddress
import os
import pathlib
import socket
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


def is_safe_url(url: str) -> bool:
    """Check if URL is safe for download (prevents basic SSRF)."""
    try:
        parsed = urlparse(url)
    except (ValueError, TypeError):
        return False
    
    if parsed.scheme not in {"http", "https"}:
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    if hostname.lower() in {"localhost", "127.0.0.1", "::1", "169.254.169.254"}:
        return False

    try:
        ip = ipaddress.ip_address(hostname)
        return not (ip.is_private or ip.is_loopback or ip.is_link_local)
    except ValueError:
        # hostname is not an IP address, need to resolve it
        pass

    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        # DNS resolution failed
        return False
    
    for info in addr_info:
        try:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except (ValueError, IndexError):
            # Skip malformed addresses
            continue
    
    return True
