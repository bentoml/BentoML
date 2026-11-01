from __future__ import annotations

import typing as t

import pytest

import bentoml._internal.client.grpc as grpc_client
from bentoml._internal.client.grpc import SyncGrpcClient


class _ChannelFactoryReached(Exception):
    pass


def _raise_when_channel_factory_is_used(server_url: str, **_: t.Any) -> t.NoReturn:
    raise _ChannelFactoryReached(server_url)


def _empty_generated_stubs(_: str) -> tuple[t.Any, t.Any]:
    return None, None


def test_sync_grpc_from_url_uses_its_channel_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CustomSyncGrpcClient(SyncGrpcClient):
        pass

    monkeypatch.setattr(
        CustomSyncGrpcClient,
        "_create_channel",
        staticmethod(_raise_when_channel_factory_is_used),
    )
    monkeypatch.setattr(grpc_client, "import_generated_stubs", _empty_generated_stubs)

    with pytest.raises(_ChannelFactoryReached) as exc_info:
        CustomSyncGrpcClient.from_url("localhost:3000")

    assert exc_info.value.args == ("0.0.0.0:3000",)


def test_sync_grpc_readiness_uses_sync_channel_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SyncGrpcClient,
        "_create_channel",
        staticmethod(_raise_when_channel_factory_is_used),
    )

    with pytest.raises(_ChannelFactoryReached) as exc_info:
        SyncGrpcClient.wait_until_server_ready("127.0.0.1", 3000)

    assert exc_info.value.args == ("127.0.0.1:3000",)
