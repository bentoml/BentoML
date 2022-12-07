from __future__ import annotations

import os
import logging
from pathlib import Path

from ..utils.pkg import source_locations
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ...grpc.utils import LATEST_PROTOCOL_VERSION
from ..configuration import is_pypi_installed_bentoml

logger = logging.getLogger(__name__)

BENTOML_DEV_BUILD = "BENTOML_BUNDLE_LOCAL_BUILD"


def build_bentoml_editable_wheel(
    target_path: str, *, _internal_protocol_version: str = LATEST_PROTOCOL_VERSION
) -> None:
    """
    This is for BentoML developers to create Bentos that contains the local bentoml
    build based on their development branch. To enable this behavior, one must
    set envar :code:`BENTOML_BUNDLE_LOCAL_BUILD=True` before building a Bento.
    """
    if str(os.environ.get(BENTOML_DEV_BUILD, False)).lower() != "true":
        return

    if is_pypi_installed_bentoml():
        # skip this entirely if BentoML is installed from PyPI
        return

    try:
        # NOTE: build.env is a standalone library,
        # different from build. However, isort sometimes
        # incorrectly re-order the imports order.
        # isort: off
        from build.env import IsolatedEnvBuilder

        from build import ProjectBuilder

        # isort: on
    except ModuleNotFoundError as e:
        raise MissingDependencyException(
            f"Environment variable '{BENTOML_DEV_BUILD}=True', which requires the 'pypa/build' package ({e}). Install development dependencies with 'pip install -r requirements/dev-requirements.txt' and try again."
        ) from None

    # Find bentoml module path
    # This will be $GIT_ROOT/src/bentoml
    module_location = source_locations("bentoml")
    if not module_location:
        raise BentoMLException("Could not find bentoml module location.")
    bentoml_path = Path(module_location)

    if not Path(
        module_location, "grpc", _internal_protocol_version, "service_pb2.py"
    ).exists():
        raise ModuleNotFoundError(
            f"Generated stubs for version {_internal_protocol_version} are missing. Make sure to run '{bentoml_path.as_posix()}/scripts/generate_grpc_stubs.sh {_internal_protocol_version}' beforehand to generate gRPC stubs."
        ) from None

    # location to pyproject.toml
    pyproject = bentoml_path.parent.parent / "pyproject.toml"

    # this is for BentoML developer to create Service containing custom development
    # branches of BentoML library, it is True only when BentoML module is installed
    # in development mode via "pip install --editable ."
    if os.path.isfile(pyproject):
        logger.info(
            "BentoML is installed in `editable` mode; building BentoML distribution with the local BentoML code base. The built wheel file will be included in the target bento."
        )
        with IsolatedEnvBuilder() as env:
            builder = ProjectBuilder(pyproject.parent)
            builder.python_executable = env.executable
            builder.scripts_dir = env.scripts_dir
            env.install(builder.build_system_requires)
            builder.build(
                "wheel", target_path, config_settings={"--global-option": "--quiet"}
            )
    else:
        logger.info(
            "Custom BentoML build is detected. For a Bento to use the same build at serving time, add your custom BentoML build to the pip packages list, e.g. `packages=['git+https://github.com/bentoml/bentoml.git@13dfb36']`"
        )
