import pytest
import tempfile
import os

from bentoml.configuration.containers import BentoMLConfiguration, BentoMLContainer
from bentoml.exceptions import BentoMLConfigException


def test_override():
    config = BentoMLConfiguration(legacy_compatibility=False)
    config.override(["api_server", "port"], 6000)
    config_dict = config.as_dict()
    assert config_dict is not None
    assert config_dict["api_server"]["port"] == 6000


def test_override_schema_violation():
    config = BentoMLConfiguration()
    with pytest.raises(BentoMLConfigException) as e:
        config.override(["api_server", "port"], "non-integer")
    assert e is not None


def test_override_nonexist_key():
    config = BentoMLConfiguration()
    with pytest.raises(BentoMLConfigException) as e:
        config.override(["non-existent", "non-existent"], 6000)
    assert e is not None


def test_override_none_value():
    config = BentoMLConfiguration(legacy_compatibility=False)
    config.override(["api_server", "port"], None)
    config_dict = config.as_dict()
    assert config_dict is not None
    assert config_dict["api_server"]["port"] == 5000


def test_override_none_key():
    config = BentoMLConfiguration(legacy_compatibility=False)
    with pytest.raises(BentoMLConfigException) as e:
        config.override(None, 6000)
    assert e is not None


def test_override_empty_key():
    config = BentoMLConfiguration(legacy_compatibility=False)
    with pytest.raises(BentoMLConfigException) as e:
        config.override([], 6000)
    assert e is not None


def test_legacy_compatibiltiy():
    config = tempfile.NamedTemporaryFile(delete=False)
    config.write(
        b"""
api_server:
  port: 0
  max_request_size: 0
marshal_server:
  request_header_flag: Null
yatai:
  url: Null
tracing:
  zipkin_api_url: Null
instrument:
  namespace: Null
"""
    )
    config.close()

    config_dict = BentoMLConfiguration(
        default_config_file=config.name,
        legacy_compatibility=True,
        validate_schema=False,
    ).as_dict()
    os.remove(config.name)
    assert config_dict is not None
    assert config_dict["api_server"]["port"] == 5000
    assert config_dict["api_server"]["max_request_size"] == 20971520
    assert config_dict["marshal_server"]["request_header_flag"] == (
        "BentoML-Is-Merged-Request"
    )
    assert config_dict["yatai"]["url"] == ""
    assert config_dict["tracing"]["zipkin_api_url"] == ""
    assert config_dict["instrument"]["namespace"] == "BENTOML"


def test_validate_schema():
    config = tempfile.NamedTemporaryFile(delete=False)
    config.write(
        b"""
invalid_key1:
  invalid_key2: Null
"""
    )
    config.close()

    with pytest.raises(BentoMLConfigException) as e:
        BentoMLConfiguration(
            default_config_file=config.name,
            validate_schema=True,
            legacy_compatibility=False,
        )

    assert e is not None
    os.remove(config.name)


def test_api_server_workers():
    container = BentoMLContainer()

    config_auto_workers = tempfile.NamedTemporaryFile(delete=False)
    config_auto_workers.write(
        b"""
api_server:
  workers: Null
"""
    )
    config_auto_workers.close()

    container.config.from_dict(
        BentoMLConfiguration(
            default_config_file=config_auto_workers.name,
            validate_schema=False,
            legacy_compatibility=False,
        ).as_dict(),
    )
    os.remove(config_auto_workers.name)
    workers = container.api_server_workers()
    assert workers is not None
    assert workers > 0

    config_manual_workers = tempfile.NamedTemporaryFile(delete=False)
    config_manual_workers.write(
        b"""
api_server:
  workers: 42
"""
    )
    config_manual_workers.close()

    container.config.from_dict(
        BentoMLConfiguration(
            default_config_file=config_manual_workers.name,
            validate_schema=False,
            legacy_compatibility=False,
        ).as_dict(),
    )
    os.remove(config_manual_workers.name)
    workers = container.api_server_workers()
    assert workers is not None
    assert workers == 42


def test_config_file_override():
    default_config_file = tempfile.NamedTemporaryFile(delete=False)
    default_config_file.write(
        b"""
key1:
  key2:
    key3: value3
    key4: value4
    key5: value5
"""
    )
    default_config_file.close()

    override_config_file = tempfile.NamedTemporaryFile(delete=False)
    override_config_file.write(
        b"""
key1:
  key2:
    key3: override3
    key5: override5
"""
    )
    override_config_file.close()

    config = BentoMLConfiguration(
        default_config_file=default_config_file.name,
        override_config_file=override_config_file.name,
        validate_schema=False,
        legacy_compatibility=False,
    ).as_dict()

    os.remove(default_config_file.name)
    os.remove(override_config_file.name)
    print(default_config_file.name)

    assert config is not None
    assert config["key1"] is not None
    assert config["key1"]["key2"] is not None
    assert config["key1"]["key2"]["key3"] == "override3"
    assert config["key1"]["key2"]["key4"] == "value4"
    assert config["key1"]["key2"]["key5"] == "override5"
