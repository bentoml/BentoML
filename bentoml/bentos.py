"""
User facing python APIs for managing local bentos and build new bentos
"""
import logging
import os
import typing as t
from typing import TYPE_CHECKING

import fs
from simple_di import Provide, inject

from ._internal.bento import Bento
from ._internal.configuration.containers import BentoMLContainer
from ._internal.service import load
from ._internal.types import Tag

if TYPE_CHECKING:  # pragma: no cover
    from ._internal.bento import BentoStore, SysPathBento
    from ._internal.models import ModelStore


logger = logging.getLogger(__name__)


@inject
def list(  # pylint: disable=redefined-builtin
    tag: t.Optional[t.Union[Tag, str]] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> "t.List[SysPathBento]":
    return _bento_store.list(tag)


@inject
def get(
    tag: t.Union[Tag, str],
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> Bento:
    return _bento_store.get(tag)


def delete(
    tag: t.Union[Tag, str],
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
):
    _bento_store.delete(tag)


def import_bento(path: str) -> Bento:
    return Bento.from_fs(fs.open_fs(path))


def export_bento(tag: t.Union[Tag, str], path: str):
    bento = get(tag)
    bento.export(path)


def push(tag: t.Union[Tag, str]):
    bento = get(tag)
    bento.push()


def pull(tag: t.Union[Tag, str]):
    raise NotImplementedError


@inject
def build(
    svc_import_str: str,
    version: t.Optional[str] = None,
    include: t.Optional[t.List[str]] = None,
    exclude: t.Optional[t.List[str]] = None,
    env: t.Optional[t.Dict[str, t.Any]] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    additional_models: t.Optional[t.List[str]] = None,
    build_ctx: t.Optional[str] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
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
    svc.set_build_options(
        include=["*.py"],
        env=dict(
            pip_install = "./requirements.txt"
        )
    )

    # For advanced build use cases, here's all the common build options:
    svc.set_build_options(
        version="any_version_label",
        models=["iris_model:latest"],
        include=['*'],
        exclude=[], # files to exclude can also be specified with a .bentoignore file
        labels={
            "foo": "bar",
        },
        env={
            "python": {
                "version": '3.7.11',
                "wheels": ["./wheels/*"],
                "pip_install": "auto",
                # "pip_install": "./requirements.txt",
                # "pip_install": ["tensorflow", "numpy"],
                # "pip_install": bentoml.build_utils.lock_pypi_version(["tensorflow", "numpy"]),
                # "pip_install": None, # do not install any python packages automatically
            },
            "docker": {
                # "base_image": "mycompany.com/registry/name/tag",
                "distro": "amazonlinux2",
                "gpu": True,
                "setup_script": "sh setup_docker_container.sh",
            },
            # "conda": "./environment.yml",
            "conda": {
                "channels": [],
                "dependencies": []
            }
        },
    )

    From CLI:

        bentoml build bento.py:svc

    Alternatively, write a python script to build new Bento with `bentoml.build` API:

        import bentoml

        if __name__ == "__main__":
            bentoml.build(
                'fraud_detector.py:svc',
                version="custom_version_str",
                additional_models=['iris_classifier:v123'],
                include=["*"],
                exclude=["*.storage", "credentials.yaml"], # + anything specified in .bentoml_ignore file
                env=dict(
                    pip_install=bentoml.utils.find_required_pypi_packages(svc),
                    conda="./environment.yaml",
                    docker={
                        "base_image": "mycompany.com/registry/name/tag",
                        "setup_script": "./setup_docker_container.sh",
                    },
                ),
                labels={
                    "team": "foo",
                    "dataset_version": "abc",
                    "framework": "pytorch",
                }
            )

    Additional build utility functions:

        from bentoml.build_utils import lock_pypi_versions
        lock_pypi_versions(["pytorch", "numpy"]) => ["pytorch==1.0", "numpy==1.23"]

        from bentoml.build_utils import with_pip_install_options
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
    """  # noqa: LN001
    build_ctx = os.getcwd() if build_ctx is None else os.path.realpath(build_ctx)
    svc = load(svc_import_str, working_dir=build_ctx)

    additional_models = [] if additional_models is None else additional_models
    include = ["*"] if include is None else include
    exclude = [] if exclude is None else exclude
    env = {} if env is None else env
    labels = {} if labels is None else labels

    bento = Bento.create(
        svc,
        build_ctx,
        additional_models,
        version,
        include,
        exclude,
        env,
        labels,
        _model_store,
    ).save(_bento_store)

    logger.info("Bento build success, %s created", bento)

    return bento


__all__ = [
    "list",
    "get",
    "delete",
    "import_bento",
    "export_bento",
    "push",
    "pull",
    "build",
]
