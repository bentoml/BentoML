import importlib
import logging
import os
import shutil
import typing as t

from simple_di import Provide, inject

import bentoml

from ..configuration import is_pip_installed_bentoml
from ..configuration.containers import BentoMLContainer
from ..utils.tempdir import TempDirectory

from .bento import Bento

if t.TYPE_CHECKING:
    from bentoml._internal.service import Service

    from ..bento.store import BentoStore
    from ..models.store import ModelStore

logger = logging.getLogger(__name__)


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
    svc: t.Union["Service", str],
    models: t.Optional[t.List[str]] = None,
    version: t.Optional[str] = None,
    description: t.Optional[str] = None,
    include: t.Optional[t.List[str]] = None,
    exclude: t.Optional[t.List[str]] = None,
    env: t.Optional[t.Dict[str, t.Any]] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Bento:
    """
    Build a Bento for this Service. A Bento is a file archive containing all the
    specifications, source code, and model files required to run and operate this
    service in production.

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
    svc.bento_files.include = ["*"]
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
        from bento import svc

        bentoml.build(
            svc,
            version="custom_version_str",
            description=open("readme.md").read(),
            models=['iris_classifier:v123'],
            include=["*"],
            exclude=["*.storage", "credentials.yaml"],
            # + anything specified in .bentoml_ignore file
            env=dict(
                pip_install=bentoml.utils.find_required_pypi_packages(svc),
                conda_environment="./environment.yaml",
                 docker_options={
                    "base_image": bentoml.utils.builtin_docker_image("slim", gpu=True),
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
    build_ctx = os.getcwd()

    if isinstance(svc, str):
        svc = bentoml.load(svc)

    res = Bento(
        svc,
        build_ctx,
        models,
        version,
        description,
        include,
        exclude,
        env,
        labels,
        model_store,
    )
    res.save(bento_store)

    return res
