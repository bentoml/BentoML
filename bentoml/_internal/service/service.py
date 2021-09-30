from typing import Any, Dict, List, Optional

from bentoml._internal.io_descriptors import IODescriptor
from bentoml._internal.utils.validation import check_is_dns1123_subdomain
from bentoml.exceptions import BentoMLException

from .inference_api import InferenceAPI
from .runner import Runner


class Service:
    """
    Default Service/Bento description goes here
    """

    _apis: Dict[str, InferenceAPI] = {}
    _runners: Dict[str, Runner] = {}

    # Name of the service, it is a required parameter for __init__
    name: str
    # Version of the service, only applicable if the service was load from a bento
    version: Optional[str] = None

    def __init__(self, name: str, runners: Optional[List[Runner]] = None):
        # Service name must be a valid dns1123 subdomain string
        check_is_dns1123_subdomain(name)
        self.name = name

        if runners is not None:
            self._runners = {r.name: r for r in runners}

    def api(
        self,
        input: IODescriptor,
        output: IODescriptor,
        api_name: Optional[str] = None,
        api_doc: Optional[str] = None,
        route: Optional[str] = None,
    ):
        """Decorate a user defined function to make it an InferenceAPI of this service"""

        def decorator(func):
            self._add_inference_api(func, input, output, api_name, api_doc, route)

        return decorator

    def _add_inference_api(
        self,
        func: callable,
        input: IODescriptor,
        output: IODescriptor,
        api_name: Optional[str],
        api_doc: Optional[str],
        route: Optional[str],
    ):
        api = InferenceAPI(
            name=api_name,
            user_defined_callback=func,
            input_descriptor=input,
            output_descriptor=output,
            doc=api_doc,
            route=route,
        )

        if api.name in self._apis:
            raise BentoMLException(
                f"API {api_name} is already defined in Service {self.name}"
            )
        self._apis[api.name] = api

    def _asgi_app(self):
        return self._app

    def _wsgi_app(self):
        return self._app

    def mount_asgi_app(self, app, path=None):
        self._app.mount(app, path=path)

    def add_middleware(self, middleware, *args, **kwargs):
        self._app

    def openapi_doc(self):
        from .openapi import get_service_openapi_doc

        return get_service_openapi_doc(self)

    def build(
        self,
        models: List[str],
        version: Optional[str] = None,
        description: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        env: Optional[Dict[str, Any]] = None,
        labels: Optional[Dict[str, str]] = None,
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
        from bentoml._internal.bento.build import build_bento

        build_bento(
            self, models, version, description, include, exclude, env, labels,
        )
