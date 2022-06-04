from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import pytest

from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import BentoMLException
from bentoml._internal.bento.gen import get_template_env
from bentoml._internal.bento.gen import expands_bento_path
from bentoml._internal.bento.gen import generate_dockerfile
from bentoml._internal.bento.gen import clean_bentoml_version
from bentoml._internal.bento.gen import validate_setup_blocks
from bentoml._internal.bento.docker import DistroSpec
from bentoml._internal.bento.docker import get_supported_spec


def test_invalid_spec():
    with pytest.raises(InvalidArgument):
        get_supported_spec("invalid_spec")


@pytest.mark.parametrize("distro", ["debian", "alpine"])
def test_distro_spec(distro: str):
    assert DistroSpec.from_distro(distro)
    assert not DistroSpec.from_distro(None)
    with pytest.raises(BentoMLException):
        DistroSpec.from_distro("invalid_distro")
