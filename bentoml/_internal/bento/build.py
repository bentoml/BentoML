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
import typing as t
import uuid

import yaml
from simple_di import Provide, inject

from bentoml.exceptions import BentoMLException, InvalidArgument

from ..configuration import is_pip_installed_bentoml
from ..configuration.containers import BentoMLContainer
from ..utils import generate_new_version_id
from ..utils.tempdir import TempDirectory

if t.TYPE_CHECKING:
    from bentoml._internal.service import Service

    from ..bento.store import BentoStore
    from ..models.store import ModelStore

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


@inject
def build_bento(
    svc: "Service",
    models: t.List[str],
    version: t.Optional[str] = None,
    description: t.Optional[str] = None,
    include: t.Optional[t.List[str]] = None,
    exclude: t.Optional[t.List[str]] = None,
    env: t.Optional[t.Dict[str, t.Any]] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
):
    """
    Build a Bento for this Service. A Bento is a file archive containing all the
    specifications, source code, models files required to run and operate this
    Service in production

    Example Usages:

    # bento.py
    import numpy as np
    import bentoml
    import bentoml.sklearn
    from bentoml.io import NumpyNdarray

    iris_model_runner = bentoml.sklearn.load_runner('iris_classifier:latest')
    svc = bentoml.Service(
        "IrisClassifier",
        runners=[iris_model_runner]
    )

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def predict(request_data: np.ndarray):
        return iris_model_runner.predict(request_data)

    # For simple use cases, only models list is required:
    svc.bento_options.models = []
    svc.bento_files.include
    svc.bento_env.pip_install = "./requirements.txt"

    # For advanced build use cases, here's all the common build options:
    @svc.build
    def build(bento_ctx):
        opts, files, env = bento_ctx

        opts.version = "custom_version_str"
        opts.description = open("readme.md").read()
        opts.models = ['iris_classifier:v123']

        files.include = ["**.py", "config.json"]
        files.exclude = ["*.pyc"]  # + anything specified in .bentoml_ignore

        env.pip_install=bentoml.utils.find_required_pypi_packages(svc)
        env.conda_environment="./environment.yaml"
        env.docker_options=dict(
            base_image=bentoml.utils.builtin_docker_image("slim", gpu=True),
            entrypoint="bentoml serve module_file:svc_name --production",
            setup_script="./setup_docker_container.sh",
        )

    # From CLI:
    bentoml build bento.py
    bentoml build bento.py:svc


    # build.py
    import bentoml

    if __name__ == "__main__":
        bentoml.build(
            "bento.py:svc",
            version="custom_version_str",
            description=open("readme.md").read(),
            models=['iris_classifier:v123'],
            include=["**.py", "config.json"]
            exclude=["*.storage"], # + anything specified in .bentoml_ignore file
            env=dict(
                pip_install=bentoml.utils.find_required_pypi_packages(svc),
                conda_environment="./environment.yaml",
                 docker_options={
                    "base_image": bentoml.utils.builtin_docker_image("slim", gpu=True)
                    "entrypoint": "bentoml serve module_file:svc_name --production",
                    "setup_script": "./setup_docker_container.sh",
                },
            ),
            labels={
                "team": "foo",
                "dataset_version": "abc",
                "framework": "pytorch",
            }
        )

    # additional env utility functions:
    from bentoml.utils import lock_pypi_versions
    lock_pypi_versions(["pytorch", "numpy"]) => ["pytorch==1.0", "numpy==1.23"]

    from bentoml.utils import with_pip_install_options
    with_pip_install_options(
          ["pytorch", "numpy"],
          index_url="https://mirror.baidu.com/pypi/simple",
          extra_index_url="https://mirror.baidu.com/pypi/simple",
          find_links="https://download.pytorch.org/whl/torch_stable.html"
     )
    > [
        "pytorch --index-url=https://mirror.baidu.com/pypi/simple --extra-index-url=https://mirror.baidu.com/pypi/simple --find-links=https://download.pytorch.org/whl/torch_stable.html",
        "numpy --index-url=https://mirror.baidu.com/pypi/simple --extra-index-url=https://mirror.baidu.com/pypi/simple --find-links=https://download.pytorch.org/whl/torch_stable.html"
    ]

    # conda dependencies:
    svc.build(
        ...
        env={
            "conda_environment": dict(
                channels=[...],
                dependencies=[...],
            )
        }
    )

    # example:

    # build.py
    from bento import svc
    from bentoml.utils import lock_pypi_versions

    if __name__ == "__main__":
        svc.build(
            models=['iris_classifier:latest'],
            include=['*'],
            env=dict(
                pip_install=lock_pypi_versions([
                    "pytorch",
                    "numpy",
                ])
            )
        )
    """
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
                model_info = model_store.get(model_tag)
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
