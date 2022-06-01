from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING
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
