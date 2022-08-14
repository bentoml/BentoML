from __future__ import annotations

import os
import logging

from ..utils.pkg import source_locations
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..configuration import is_pypi_installed_bentoml

logger = logging.getLogger(__name__)

BENTOML_DEV_BUILD = "BENTOML_BUNDLE_LOCAL_BUILD"
_exc_message = f"'{BENTOML_DEV_BUILD}=True', which requires the 'pypa/build' package. Install development dependencies with 'pip install -r requirements/dev-requirements.txt' and try again."


def build_bentoml_editable_wheel(target_path: str) -> None:
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
        from build.env import IsolatedEnvBuilder

        from build import ProjectBuilder
    except ModuleNotFoundError as e:
        raise MissingDependencyException(_exc_message) from e

    # Find bentoml module path
    module_location = source_locations("bentoml")
    if not module_location:
        raise BentoMLException("Could not find bentoml module location.")

    try:
        from importlib import import_module

        _ = import_module("bentoml.grpc.v1alpha1.service_pb2")
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Generated stubs are not found. Make sure to run '{module_location}/scripts/generate_grpc_stubs.sh' beforehand to generate gRPC stubs."
        ) from None

    pyproject = os.path.abspath(os.path.join(module_location, "..", "pyproject.toml"))

    # this is for BentoML developer to create Service containing custom development
    # branches of BentoML library, it is True only when BentoML module is installed
    # in development mode via "pip install --editable ."
    if os.path.isfile(pyproject):
        logger.info(
            "BentoML is installed in `editable` mode; building BentoML distribution with the local BentoML code base. The built wheel file will be included in the target bento."
        )
        with IsolatedEnvBuilder() as env:
            builder = ProjectBuilder(os.path.dirname(pyproject))
            builder.python_executable = env.executable
            builder.scripts_dir = env.scripts_dir
            env.install(builder.build_system_requires)
            builder.build("wheel", target_path)
    else:
        logger.info(
            "Custom BentoML build is detected. For a Bento to use the same build at serving time, add your custom BentoML build to the pip packages list, e.g. `packages=['git+https://github.com/bentoml/bentoml.git@13dfb36']`"
        )
