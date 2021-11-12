"""
User facing python APIs for managing local bentos and build new bentos
"""
import logging
import os
import typing as t

import fs
from simple_di import Provide, inject

from ._internal.bento import Bento, BentoStore
from ._internal.configuration.containers import BentoMLContainer
from ._internal.service import load
from ._internal.types import Tag

if t.TYPE_CHECKING:
    from ._internal.models.store import ModelStore


logger = logging.getLogger(__name__)


@inject
def list(
    tag: t.Optional[t.Union[Tag, str]] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> t.List[Bento]:
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
    # FIXME: find bento tag from path
    return Bento.from_fs("TODO", fs.open_fs(path))


def export_bento(tag: t.Union[Tag, str], path: str):
    bento = get(tag)
    bento.export(path)


def push(tag: t.Union[Tag, str]):
    bento = get(tag)
    bento.push()


def pull(tag: t.Union[Tag, str]):
    pass


@inject
def build(
    svc_import_str: str,
    models: t.Optional[t.List[str]] = None,
    version: t.Optional[str] = None,
    description: t.Optional[str] = None,
    include: t.Optional[t.List[str]] = None,
    exclude: t.Optional[t.List[str]] = None,
    env: t.Optional[t.Dict[str, t.Any]] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
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
        description=open("README.md").read(),
        models=["iris_model:latest"],
        include=['*'],
        exclude=[], # files to exclude can also be specified with a .bentomlignore file
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
                description=open("readme.md").read(),
                models=['iris_classifier:v123'],
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

    description = svc.__doc__ if description is None else description
    models = [] if models is None else models
    include = ["*"] if include is None else include
    exclude = [] if exclude is None else exclude
    env = {} if env is None else env
    labels = {} if labels is None else labels

    bento = Bento.create(
        svc,
        build_ctx,
        models,
        version,
        description,
        include,
        exclude,
        env,
        labels,
        _model_store,
    ).save(_bento_store)

    logger.info("%s created at: %s", bento, bento.fs.getsyspath("/"))

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
