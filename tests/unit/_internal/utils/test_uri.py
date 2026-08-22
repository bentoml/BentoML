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


def _seed_fake_original(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the 'original' create_connection so blocked targets never dial out."""
    from bentoml._internal.utils import uri

    attempts: list[str] = []

    async def fake_original(self, protocol_factory, host=None, port=None, **kwargs):  # type: ignore[no-untyped-def]
        assert host is not None
        attempts.append(host)
        return object()  # sentinel; the real connect is never reached

    monkeypatch.setattr(uri, "original_create_connection", fake_original, raising=False)
    return attempts


@pytest.mark.asyncio
async def test_safe_connect_blocks_cgnat_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """100.64.0.0/10 (RFC 6598 CGNAT) must be rejected (#5644)."""
    import socket

    import uvloop

    from bentoml._internal.utils import uri

    _seed_fake_original(monkeypatch)
    with uri.make_safe_connect(), pytest.raises(socket.gaierror):
        await uvloop.Loop.create_connection(
            None,
            None,
            host="100.64.1.1",
            port=80,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("host", "blocked"),
    [
        # RFC 6598 boundaries: block exactly 100.64.0.0 - 100.127.255.255.
        ("100.63.255.255", False),
        ("100.64.0.0", True),
        ("100.96.0.1", True),
        ("100.127.255.255", True),
        ("100.128.0.0", False),
        # IPv4-mapped IPv6 literals resolve to their embedded IPv4 address.
        ("::ffff:100.64.1.1", True),
        ("::ffff:192.168.0.1", True),
        ("::ffff:8.8.8.8", False),
        # Pre-existing CVE-2025-54381 behavior still holds.
        ("192.168.0.1", True),
        ("10.0.0.7", True),
        ("127.0.0.1", True),
        ("169.254.169.254", True),
        # Public addresses pass through to the underlying connect.
        ("8.8.8.8", False),
        # Reserved ranges are now covered too.
        ("240.0.0.1", True),
    ],
)
@pytest.mark.asyncio
async def test_safe_connect_target_policy(
    host: str, blocked: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    import uvloop

    from bentoml._internal.utils import uri

    attempts = _seed_fake_original(monkeypatch)
    with uri.make_safe_connect():
        if blocked:
            with pytest.raises(socket.gaierror):
                await uvloop.Loop.create_connection(
                    None,
                    None,
                    host=host,
                    port=80,  # type: ignore[arg-type]
                )
        else:
            await uvloop.Loop.create_connection(
                None,
                None,
                host=host,
                port=80,  # type: ignore[arg-type]
            )
        assert (host in attempts) is not blocked
