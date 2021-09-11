import datetime
import glob
import gzip
import importlib
import io
import json
import logging
import os
import re
import shutil
import stat
import tarfile
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

from bentoml._internal.configuration import is_pip_installed_bentoml
from bentoml._internal.models.stores import modelstore
from bentoml._internal.utils import generate_new_version_id
from bentoml._internal.utils.tempdir import TempDirectory
from bentoml.exceptions import BentoMLException, InvalidArgument

from .store import bento_store

if TYPE_CHECKING:
    from bentoml._internal.service import Service

logger = logging.getLogger(__name__)


def validate_version_str(version_str):
    """
    Validate that version str format is either a simple version string that:
        * Consist of only ALPHA / DIGIT / "-" / "." / "_"
        * Length between 1-128
    Or a valid semantic version https://github.com/semver/semver/blob/master/semver.md
    """
    regex = r"[A-Za-z0-9_.-]{1,128}\Z"
    semver_regex = r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"  # noqa: E501
    if (
        re.match(regex, version_str) is None
        and re.match(semver_regex, version_str) is None
    ):
        raise InvalidArgument(
            'Invalid Service version: "{}", it can only consist'
            ' ALPHA / DIGIT / "-" / "." / "_", and must be less than'
            "128 characters".format(version_str)
        )

    if version_str.lower() == "latest":
        raise InvalidArgument('Service version can not be set to "latest"')


def build_bentoml_whl_to_target_if_in_editable_mode(target_path):
    """
    if bentoml is installed in editor mode(pip install -e), this will build a wheel
    distribution with the local bentoml source and add it to saved bento directory
    under {bento_path}/env/python/wheels/
    """
    if is_pip_installed_bentoml():
        # skip this entirely if BentoML is installed from PyPI
        return

    # Find bentoml module path
    (module_location,) = importlib.util.find_spec("bentoml").submodule_search_locations

    bentoml_setup_py = os.path.abspath(os.path.join(module_location, "..", "setup.py"))

    # this is for BentoML developer to create Service containing custom development
    # branches of BentoML library, it is True only when BentoML module is installed
    # in development mode via "pip install --editable ."
    if os.path.isfile(bentoml_setup_py):
        logger.warning(
            "BentoML is installed in `editable` model, building BentoML distribution "
            "with local BentoML code base. The built wheel file will be included in"
            "target bento directory, under: {bento_path}/env/python/wheels/"
        )

        with TempDirectory() as tempdir:
            from setuptools import sandbox

            # build BentoML wheel distribution under tempdir
            sandbox.run_setup(
                bentoml_setup_py,
                ["-q", "bdist_wheel", "--dist-dir", tempdir],
            )

            # copy the built wheel file to target directory
            shutil.copytree(tempdir, target_path)


def build_bento(
    svc: Service,
    models: List[str],
    version: Optional[str] = None,
    description: Optional[str] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    env: Optional[Dict[str, Any]] = None,
    labels: Optional[Dict[str, str]] = None,
):
    if version is None:
        version = generate_new_version_id()

    validate_version_str(version)

    bento_tag = f"{svc.name}:{version}"
    build_ctx = os.getcwd()
    logger.debug(f"Building BentoML service {bento_tag} from build context {build_ctx}")

    with bento_store.register_bento(bento_tag) as bento_path:
        # Copy required models from local modelstore, into
        # `models/{model_name}/{model_version}` directory
        for model_tag in models:
            try:
                model_info = modelstore.get_model(model_tag)
            except FileNotFoundError:
                raise BentoMLException(
                    f"Model {model_tag} not found in local model store"
                )

            model_name, model_version = model_tag.split(":")
            target_path = os.path.join(bento_path, "models", model_name, model_version)
            shutil.copytree(model_info.path, target_path)

        # Copy all files base on include and exclude, into `{svc.name}` directory
        # TODO

        # Create env, docker, bentoml dev whl files
        # TODO

        # Create `readme.md` file
        description = svc.__doc__ if description is None else description
        readme_path = os.path.join(bento_path, "readme.md")
        with open(readme_path, "w") as f:
            f.write(description)

        # Create 'api/openapi.yaml' file
        api_docs_path = os.path.join(bento_path, "apis")
        os.mkdir(api_docs_path)
        openapi_docs_file = os.path.join(api_docs_path, "openapi.yaml")
        with open(openapi_docs_file, "w") as f:
            yaml.safe_dump(svc.openapi_doc, f)

        # Create bento.yaml
        # TODO
