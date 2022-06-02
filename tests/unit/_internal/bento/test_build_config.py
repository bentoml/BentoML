from __future__ import annotations

import sys
import typing as t
import logging
from typing import TYPE_CHECKING
from unittest import TestCase
from unittest.mock import patch

import pytest

from bentoml._internal.bento.gen import get_template_env
from bentoml._internal.bento.gen import generate_dockerfile
from bentoml._internal.bento.gen import validate_setup_blocks
from bentoml._internal.bento.docker import DistroSpec
from bentoml._internal.bento.build_config import CondaOptions
from bentoml._internal.bento.build_config import DockerOptions
from bentoml._internal.bento.build_config import PythonOptions
from bentoml._internal.bento.build_config import BentoBuildConfig

if TYPE_CHECKING:
    from unittest.mock import MagicMock


@patch("bentoml._internal.bento.build_config.logger")
def test_warning_logs(mock_logger: MagicMock):
    with patch.object(sys, "version_info") as pyver:
        pyver.major = 3
        pyver.minor = 2
        print(pyver)

        print(bentoml._internal.bento.build_config.PYTHON_VERSION)
        mock_logger.warning.assert_called()


class TestDockerOptions(TestCase):
    def setUp(self) -> None:
        ...
