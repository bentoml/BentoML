import ipaddress
import os
import typing as t

import psutil
import pytest

WINDOWS_PATHS = [
    r"C:\foo\bar",
    r"C:\foo\bar with space",
    r"C:\\foo\\中文",
    r"relative\path",
    # r"\\localhost\c$\WINDOWS\network",
    # r"\\networkstorage\homes\user",
]
POSIX_PATHS = ["/foo/bar", "/foo/bar with space", "/foo/中文", "relative/path"]


@pytest.fixture()
def example_paths():
    if psutil.WINDOWS:
        return WINDOWS_PATHS
    else:
        return POSIX_PATHS


def test_uri_path_conversion(
    example_paths: t.List[str],  # pylint: disable=redefined-outer-name
) -> None:
    from bentoml._internal.utils.uri import path_to_uri
    from bentoml._internal.utils.uri import uri_to_path

    for path in example_paths:
        restored = uri_to_path(path_to_uri(path))
        assert restored == path or restored == os.path.abspath(path)


@pytest.mark.parametrize(
    "host",
    [
        "100.64.0.1",  # CGNAT lower bound
        "100.100.0.1",  # CGNAT (the reported SSRF bypass, GHSA-mrmq-3q62-6cc8 follow-up)
        "100.127.255.255",  # CGNAT upper bound
        "192.168.0.1",  # private
        "10.0.0.1",  # private
        "127.0.0.1",  # loopback
        "169.254.1.1",  # link-local
        "240.0.0.1",  # reserved
        "::1",  # IPv6 loopback
        "fc00::1",  # IPv6 unique local (private)
        "::ffff:100.64.1.1",  # IPv4-mapped IPv6 CGNAT
        "::ffff:100.100.0.1",  # IPv4-mapped IPv6 CGNAT
        "::ffff:127.0.0.1",  # IPv4-mapped IPv6 loopback
        "::ffff:192.168.1.1",  # IPv4-mapped IPv6 private
    ],
)
def test_is_unsafe_address_blocks_internal_ranges(host: str) -> None:
    from bentoml._internal.utils.uri import is_unsafe_address

    assert is_unsafe_address(ipaddress.ip_address(host)) is True


@pytest.mark.parametrize(
    "host",
    [
        "100.63.255.255",  # just below the CGNAT range
        "100.128.0.1",  # just above the CGNAT range
        "8.8.8.8",  # public
        "1.1.1.1",  # public
        "2001:4860:4860::8888",  # public IPv6
        "::ffff:8.8.8.8",  # IPv4-mapped IPv6 public
    ],
)
def test_is_unsafe_address_allows_public_addresses(host: str) -> None:
    from bentoml._internal.utils.uri import is_unsafe_address

    assert is_unsafe_address(ipaddress.ip_address(host)) is False
