from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import psutil
import pytest

from bentoml._internal.server.grpc import Config
from bentoml._internal.server.grpc import Servicer

if TYPE_CHECKING:
    from bentoml import Service


@pytest.fixture()
def servicer(simple_service: Service) -> Servicer:
    return Servicer(simple_service)


@pytest.mark.skipif(not psutil.WINDOWS, reason="Windows test.")
def test_windows_config_options(servicer: Servicer) -> None:
    config = Config(
        servicer,
        bind_address="0.0.0.0",
        max_message_length=None,
        max_concurrent_streams=None,
        maximum_concurrent_rpcs=None,
    )
    assert not config.options


@pytest.mark.skipif(psutil.WINDOWS, reason="Unix test.")
@pytest.mark.parametrize(
    "options,expected",
    [
        (
            {"max_concurrent_streams": 128},
            (
                ("grpc.so_reuseport", 1),
                ("grpc.max_concurrent_streams", 128),
                ("grpc.max_message_length", -1),
                ("grpc.max_receive_message_length", -1),
                ("grpc.max_send_message_length", -1),
            ),
        ),
        (
            {"max_message_length": 2048},
            (
                ("grpc.so_reuseport", 1),
                ("grpc.max_message_length", 2048),
                ("grpc.max_receive_message_length", 2048),
                ("grpc.max_send_message_length", 2048),
            ),
        ),
    ],
)
def test_unix_options(
    servicer: Servicer,
    options: dict[str, t.Any],
    expected: tuple[tuple[str, t.Any], ...],
) -> None:
    config = Config(servicer, bind_address="0.0.0.0", **options)
    assert config.options
    assert config.options == expected
