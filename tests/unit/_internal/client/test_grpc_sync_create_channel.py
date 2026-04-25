"""Regression test for #4263 / #4683: SyncGrpcClient mistakenly calls
``GrpcClient._create_channel`` (which doesn't exist), instead of its own
``SyncGrpcClient._create_channel``.

Before the fix, both ``SyncGrpcClient.from_url`` and
``SyncGrpcClient.wait_until_server_ready`` raised
``AttributeError: type object 'GrpcClient' has no attribute '_create_channel'``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("grpc")


def test_sync_grpc_client_has_create_channel() -> None:
    # The wrapper class ``GrpcClient`` does NOT define ``_create_channel`` —
    # only ``SyncGrpcClient`` and ``AsyncGrpcClient`` do. Sync code paths
    # MUST resolve ``_create_channel`` via ``SyncGrpcClient``.
    from bentoml._internal.client.grpc import GrpcClient
    from bentoml._internal.client.grpc import SyncGrpcClient

    assert not hasattr(GrpcClient, "_create_channel"), (
        "GrpcClient is a thin wrapper and intentionally has no "
        "_create_channel. Sync paths must use SyncGrpcClient._create_channel."
    )
    assert hasattr(SyncGrpcClient, "_create_channel")


def test_sync_grpc_client_create_channel_returns_sync_channel() -> None:
    # ``SyncGrpcClient._create_channel`` must return a *sync* channel that
    # supports ``with ... as channel``. Using the async ``aio.insecure_channel``
    # returns a channel that only supports ``async with``, breaking sync paths.
    from bentoml._internal.client.grpc import SyncGrpcClient

    channel = SyncGrpcClient._create_channel("127.0.0.1:65530")
    try:
        assert hasattr(channel, "__enter__"), (
            "SyncGrpcClient._create_channel must return a sync context manager."
        )
    finally:
        # ``grpc.insecure_channel`` exposes ``close`` synchronously.
        channel.close()


def test_sync_grpc_client_wait_until_server_ready_no_attribute_error() -> None:
    # Regression for #4683: ``wait_until_server_ready`` was calling
    # ``GrpcClient._create_channel`` (which doesn't exist) and crashed with
    # AttributeError before it ever attempted a connection.
    from bentoml._internal.client.grpc import SyncGrpcClient

    with pytest.raises(Exception) as excinfo:
        SyncGrpcClient.wait_until_server_ready("127.0.0.1", 65531, timeout=1)
    # Whatever happens, it must NOT be an AttributeError on _create_channel.
    assert not (
        isinstance(excinfo.value, AttributeError)
        and "_create_channel" in str(excinfo.value)
    ), f"Regression: {excinfo.value!r}"


def test_sync_grpc_client_from_url_no_attribute_error(monkeypatch) -> None:
    # Regression for #4683: ``SyncGrpcClient.from_url`` crashed with
    # ``AttributeError: type object 'GrpcClient' has no attribute
    # '_create_channel'`` because it called the wrong class's classmethod.
    from bentoml._internal.client.grpc import SyncGrpcClient

    with pytest.raises(Exception) as excinfo:
        SyncGrpcClient.from_url("127.0.0.1:65532")
    assert not (
        isinstance(excinfo.value, AttributeError)
        and "_create_channel" in str(excinfo.value)
    ), f"Regression: {excinfo.value!r}"
