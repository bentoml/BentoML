"""
Bento is a standardized file archive format in BentoML, which describes how to setup
and run a `bentoml.Service` defined by a user. It includes user code which instantiates
the `bentoml.Service` instance, as well as related configurations, data/model files,
and dependencies.

Here's an example file structure of a Bento

    /example_bento
     - readme.md
     - bento.yml
     - /apis/
         - openapi.yaml # openapi spec
         - proto.pb # Note: gRPC proto is not currently available
     - /env/
         - /python
             - python_version.txt
             - requirements.txt
             - /wheels
         - /docker
             - Dockerfile
             - Dockerfile-gpu  # optional
             - Dockerfile-foo  # optional
             - docker-entrypoint.sh
           - bentoml-init.sh
           - setup-script  # optional
         - /conda
             - environment.yml

     - /FraudDetector  # this folder is mostly identical to user's development directory
        - bento.py
        - /common
           - my_lib.py
        - my_config.json

     - /models
        - /my_nlp_model
           - bentoml_model.yml
           - model.pkl


An example `bento.yml` file in a Bento directory:

    service: bento:svc
    name: FraudDetector
    version: 20210709_DE14C9
    bentoml_version: 1.1.0

    created_at: ...

    labels:
        foo: bar
        abc: def
        author: parano
        team: bentoml

    apis:
    - predict:  # api_name is the key
        route: ...
        docs: ...
        input:
            type: bentoml.io.PandasDataFrame
            options:
                orient: "column"
        output:
            type: bentoml.io.JSON
            options:
                schema:
                    # pydantic model .schema() here

    models:
    - my_nlp_model:20210709_C154BA



Build docker image from a Bento directory:

    cd bento_path
    docker build -f ./env/docker/Dockerfile .
"""
import os
import typing as t

import fs
from fs.mirror import mirror
from simple_di import Provide, inject

from ._internal.bento import Bento
from ._internal.configuration.containers import BentoMLContainer
from ._internal.types import BentoTag

if t.TYPE_CHECKING:
    from ._internal.models.store import ModelStore
    from ._internal.service import Service
    from ._internal.store import Store

from ._internal.service import load


@inject
def list(
    tag: t.Optional[t.Union[BentoTag, str]] = None,
    bento_store: "Store" = Provide[BentoMLContainer.bento_store],
) -> t.List[BentoTag]:
    return bento_store.list(tag)


@inject
def get(
    tag: t.Union[BentoTag, str],
    bento_store: "Store" = Provide[BentoMLContainer.bento_store],
) -> Bento:
    bento_fs = bento_store.get(tag)
    return Bento.import_from_fs(bento_fs)


def delete(
    tag: t.Union[BentoTag, str],
    bento_store: "Store" = Provide[BentoMLContainer.bento_store],
):
    bento_store.delete(tag)


def import_bento(path: str) -> Bento:
    return Bento.import_from_fs(fs.open_fs(path))


def export_bento(bento: Bento, path: str):
    mirror(bento.fs, fs.open_fs(path), copy_if_newer=False)
    pass


def load_runner(tag: t.Union[BentoTag, str]) -> ...:
    pass


@inject
def build(
    svc: t.Union["Service", str],
    models: t.List[str] = [],
    version: t.Optional[str] = None,
    description: t.Optional[str] = None,
    include: t.List[str] = ["*"],
    exclude: t.List[str] = [],
    env: t.Optional[t.Dict[str, t.Any]] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    bento_store: "Store" = Provide[BentoMLContainer.bento_store],
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
        svc = load(svc)

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
        model_store,
    )
    res.save(bento_store)

    return res


__all__ = [
    "list",
    "get",
    "delete",
    "import_bento",
    "export_bento",
    "build",
    "load",
]
