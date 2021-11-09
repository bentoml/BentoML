"""
User facing python APIs for managing local bentos and build new bentos
"""
import os
import typing as t

import fs
from simple_di import Provide, inject

from ._internal.bento import Bento, BentoStore
from ._internal.configuration.containers import BentoMLContainer
from ._internal.service import Service, load
from ._internal.types import Tag
from ._internal.utils import generate_new_version_id

if t.TYPE_CHECKING:
    from ._internal.models.store import ModelStore


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
    return Bento.from_fs(fs.open_fs(path))


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
    svc: t.Union["Service", str],
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
                conda="./environment.yaml",
                docker={
                    "base_image": bentoml.build_utils.builtin_docker_image("slim"),
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
    from bentoml.build_utils import lock_pypi_versions
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

    # Configuring a different docker base image:
    env = dict(
        docker={
            distro=
        }
    )

    # conda dependencies:
    svc.build(
        ...
        env={
            "conda": dict(
                channels=[...],
                dependencies=[...],
            )
        }
    )

    # example:

    # build.py
    from bento import svc
    from bentoml.build_utils import lock_pypi_versions

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
    """  # noqa: LN001
    version = generate_new_version_id() if version is None else version
    description = svc.__doc__ if description is None else description
    models = [] if models is None else models
    include = ["*"] if include is None else include
    exclude = [] if exclude is None else exclude
    env = {} if env is None else env
    labels = {} if labels is None else labels
    build_ctx = os.getcwd() if build_ctx is None else build_ctx

    if isinstance(svc, str):
        svc = load(svc)

    if isinstance(svc, Service):
        # TODO: figure out import module when Service object was imported
        assert svc._import_str is not None, "TODO - support build on imported service"

    res = Bento.create(
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
    )
    res.save(_bento_store)

    return res


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
