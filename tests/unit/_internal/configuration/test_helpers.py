from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from bentoml.exceptions import BentoMLConfigException
from bentoml._internal.configuration.helpers import flatten_dict
from bentoml._internal.configuration.helpers import rename_fields
from bentoml._internal.configuration.helpers import load_config_file
from bentoml._internal.configuration.helpers import is_valid_ip_address

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.logging import LogCaptureFixture


def test_flatten_dict():
    assert dict(flatten_dict({"a": 1, "b": {"c": 2, "d": {"e": 3}}})) == {
        "a": 1,
        "b.c": 2,
        "b.d.e": 3,
    }

    assert dict(
        flatten_dict({"runners": {"iris_clf": {"nvidia.com/gpu": [0, 1]}}})
    ) == {'runners.iris_clf."nvidia.com/gpu"': [0, 1]}

    assert dict(flatten_dict({"a": 1, "b": 2}, sep="_")) == {"a": 1, "b": 2}


def test_rename_fields(caplog: LogCaptureFixture):
    # first, if given field is not in the dictionary, nothing will happen
    d = {"a": 1, "b": 2}
    rename_fields(d, "c", "d")
    assert "a" in d

    # second, if given field is in the dictionary, it will be renamed
    d = {"api_server.port": 5000}
    with caplog.at_level(logging.WARNING):
        rename_fields(d, "api_server.port", "api_server.http.port")
    assert (
        "Field 'api_server.port' is deprecated and has been renamed to 'api_server.http.port'"
        in caplog.text
    )
    assert "api_server.http.port" in d and d["api_server.http.port"] == 5000
    caplog.clear()

    # third, if given field is in the dictionary, and remove_only is True, it will be
    # removed.
    d = {"api_server.port": 5000}
    with caplog.at_level(logging.WARNING):
        rename_fields(d, "api_server.port", remove_only=True)

    assert "Field 'api_server.port' is deprecated and will be removed." in caplog.text
    assert len(d) == 0 and "api_server.port" not in d
    caplog.clear()

    with pytest.raises(AssertionError, match="'replace_with' must be provided."):
        # fourth, if no replace_with field is given, an AssertionError will be raised
        d = {"api_server.port": 5000}
        rename_fields(d, "api_server.port")
    with pytest.raises(ValueError, match="Given dictionary is not flattened. *"):
        # fifth, if the given dictionary is not flattened, a ValueError will be raised
        d = {"a": 1, "b": {"c": 2}}
        rename_fields(d, "b.c", "b.d.c")


def test_load_config_file(tmp_path: Path):
    config = tmp_path / "configuration.yaml"
    config.write_text("api_server:\n  port: 5000")
    assert load_config_file(config.__fspath__()) == {"api_server": {"port": 5000}}

    with pytest.raises(BentoMLConfigException) as e:
        load_config_file("/tmp/nonexistent.yaml")
    assert "Configuration file /tmp/nonexistent.yaml not found." in str(e.value)


def test_valid_ip_address():
    assert is_valid_ip_address("0.0.0.0")
    assert not is_valid_ip_address("asdfadsf:143")
