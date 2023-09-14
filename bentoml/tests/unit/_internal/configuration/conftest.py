from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import pytest

from bentoml._internal.configuration.containers import BentoMLConfiguration

if TYPE_CHECKING:
    from pathlib import Path

    from simple_di.providers import ConfigDictType


@pytest.fixture(scope="function", name="container_from_file")
def fixture_container_from_file(tmp_path: Path) -> t.Callable[[str], ConfigDictType]:
    def inner(config: str) -> ConfigDictType:
        path = tmp_path / "configuration.yaml"
        path.write_text(config)
        return BentoMLConfiguration(override_config_file=path.__fspath__()).to_dict()

    return inner


@pytest.fixture(scope="function", name="container_from_envvar")
def fixture_container_from_envvar():
    def inner(override: str) -> ConfigDictType:
        return BentoMLConfiguration(override_config_values=override).to_dict()

    return inner
