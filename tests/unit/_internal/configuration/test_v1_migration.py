from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import pytest

from bentoml.exceptions import BentoMLConfigException

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from simple_di.providers import ConfigDictType


@pytest.mark.usefixtures("container_from_file")
def test_backward_configuration(
    container_from_file: t.Callable[[str], ConfigDictType],
    caplog: LogCaptureFixture,
):
    OLD_CONFIG = """\
api_server:
  max_request_size: 8624612341
  port: 5000
  host: 0.0.0.0
"""
    with caplog.at_level(logging.WARNING):
        bentoml_cfg = container_from_file(OLD_CONFIG)
    assert all(
        i not in bentoml_cfg["api_server"] for i in ("max_request_size", "port", "host")
    )
    assert bentoml_cfg["api_server"]["http"]["host"] == "0.0.0.0"
    assert bentoml_cfg["api_server"]["http"]["port"] == 5000
    assert (
        "Field 'api_server.max_request_size' is deprecated and will be removed."
        in caplog.text
    )


@pytest.mark.usefixtures("container_from_envvar")
def test_backward_from_envvar(
    container_from_envvar: t.Callable[[str], ConfigDictType],
    caplog: LogCaptureFixture,
):
    envvar = 'version=1 tracing.type="jaeger" tracing.jaeger.address="localhost" tracing.jaeger.port=6831 tracing.sample_rate=0.7'
    with caplog.at_level(logging.WARNING):
        bentoml_cfg = container_from_envvar(envvar)
    assert bentoml_cfg["version"] == 1
    assert bentoml_cfg["tracing"]["exporter_type"] == "jaeger"

    assert (
        "Field 'tracing.jaeger.address' is deprecated and has been renamed to 'tracing.jaeger.thrift.agent_host_name'"
        in caplog.text
    )


@pytest.mark.usefixtures("container_from_file")
def test_validate(container_from_file: t.Callable[[str], ConfigDictType]):
    INVALID_CONFIG = """\
api_server:
  host: localhost
"""
    with pytest.raises(
        BentoMLConfigException, match="Invalid configuration file was given:*"
    ):
        container_from_file(INVALID_CONFIG)


@pytest.mark.usefixtures("container_from_file")
def test_backward_warning(
    container_from_file: t.Callable[[str], ConfigDictType],
    caplog: LogCaptureFixture,
):
    OLD_HOST = """\
api_server:
  host: 0.0.0.0
"""
    with caplog.at_level(logging.WARNING):
        container_from_file(OLD_HOST)
    assert "Field 'api_server.host' is deprecated" in caplog.text
    caplog.clear()

    OLD_PORT = """\
api_server:
  port: 4096
"""
    with caplog.at_level(logging.WARNING):
        container_from_file(OLD_PORT)
    assert "Field 'api_server.port' is deprecated" in caplog.text
    caplog.clear()

    OLD_CORS = """\
api_server:
  cors:
    enabled: false
"""
    with caplog.at_level(logging.WARNING):
        container_from_file(OLD_CORS)
    assert "Field 'api_server.cors.enabled' is deprecated" in caplog.text
    caplog.clear()

    OLD_JAEGER_ADDRESS = """\
tracing:
    type: jaeger
    jaeger:
        address: localhost
"""
    with caplog.at_level(logging.WARNING):
        container_from_file(OLD_JAEGER_ADDRESS)
    assert "Field 'tracing.jaeger.address' is deprecated" in caplog.text
    caplog.clear()

    OLD_JAEGER_PORT = """\
tracing:
  type: jaeger
  jaeger:
    port: 6881
"""
    with caplog.at_level(logging.WARNING):
        container_from_file(OLD_JAEGER_PORT)
    assert "Field 'tracing.jaeger.port' is deprecated" in caplog.text
    caplog.clear()

    OLD_ZIPKIN_URL = """\
tracing:
  type: zipkin
  zipkin:
    url: localhost:6881
"""
    with caplog.at_level(logging.WARNING):
        container_from_file(OLD_ZIPKIN_URL)
    assert (
        "Field 'tracing.zipkin.url' is deprecated and has been renamed to 'tracing.zipkin.endpoint'"
        in caplog.text
    )
    caplog.clear()

    OLD_OTLP_URL = """\
tracing:
  type: otlp
  otlp:
    url: localhost:6881
"""
    with caplog.at_level(logging.WARNING):
        container_from_file(OLD_OTLP_URL)
    assert (
        "Field 'tracing.otlp.url' is deprecated and has been renamed to 'tracing.otlp.endpoint'"
        in caplog.text
    )
    caplog.clear()
