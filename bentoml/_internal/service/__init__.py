# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime
from typing import List

from simple_di import Provide, inject

from bentoml.adapters import BaseInputAdapter, BaseOutputAdapter, DefaultOutput
from bentoml.configuration.containers import BentoMLContainer
from bentoml.exceptions import BentoMLException, InvalidArgument, NotFound
from bentoml.saved_bundle import save_to_dir
from bentoml.saved_bundle.config import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MAX_LATENCY,
    SavedBundleConfig,
)
from bentoml.saved_bundle.pip_pkg import seek_pip_packages
from bentoml.service.artifacts import ArtifactCollection, BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv
from bentoml.service.inference_api import InferenceAPI


BENTOML_RESERVED_API_NAMES = [
    "index",
    "swagger",
    "docs",
    "healthz",
    "metrics",
    "feedback",
]
logger = logging.getLogger(__name__)
prediction_logger = logging.getLogger("bentoml.prediction")


def validate_inference_api_name(api_name: str):
    if not api_name.isidentifier():
        raise InvalidArgument(
            "Invalid API name: '{}', a valid identifier may only contain letters,"
            " numbers, underscores and not starting with a number.".format(api_name)
        )

    if api_name in BENTOML_RESERVED_API_NAMES:
        raise InvalidArgument(
            "Reserved API name: '{}' is reserved for infra endpoints".format(api_name)
        )


def validate_inference_api_route(route: str):
    if re.findall(
        r"[?#]+|^(//)|^:", route
    ):  # contains '?' or '#' OR  start with '//' OR start with ':'
        # https://tools.ietf.org/html/rfc3986#page-22
        raise InvalidArgument(
            "The path {} contains illegal url characters".format(route)
        )
    if route in BENTOML_RESERVED_API_NAMES:
        raise InvalidArgument(
            "Reserved API route: '{}' is reserved for infra endpoints".format(route)
        )


def api_decorator(
    *args,
    input: BaseInputAdapter = None,
    output: BaseOutputAdapter = None,
    api_name: str = None,
    route: str = None,
    api_doc: str = None,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    max_latency: int = DEFAULT_MAX_LATENCY,
    batch=False,
    **kwargs,
):  # pylint: disable=redefined-builtin
    """
    A decorator exposed as `bentoml.api` for defining Inference API in a BentoService
    class.

    :param input: InputAdapter instance of the inference API
    :param output: OutputAdapter instance of the inference API
    :param api_name: API name, default to the user-defined callback function's function
        name
    :param route: Specify HTTP URL route of this inference API. By default,
        `api.name` is used as the route.  This parameter can be used for customizing
        the URL route, e.g. `route="/api/v2/model_a/predict"`
        Default: None (the same as api_name)
    :param api_doc: user-facing documentation of the inference API. default to the
        user-defined callback function's docstring
    :param max_batch_size: The maximum size of requests batch accepted by this
        inference API. This parameter governs the throughput/latency trade off, and
        avoids having large batches that exceed some resource constraint (e.g. GPU
        memory to hold the entire batch's data). Default: 1000.
    :param max_latency: The latency goal of this inference API in milliseconds.
        Default: 10000.


    Example usage:

    >>> # add example usage
    """

    def decorator(func):
        _api_name = func.__name__ if api_name is None else api_name
        _api_route = _api_name if route is None else route
        validate_inference_api_name(_api_name)
        validate_inference_api_route(_api_route)
        _api_doc = func.__doc__ if api_doc is None else api_doc

        if input is None:
            # Raise error when input adapter class passed without instantiation
            if not args or not (
                inspect.isclass(args[0]) and issubclass(args[0], BaseInputAdapter)
            ):
                raise InvalidArgument(
                    "BentoService @api decorator first parameter must "
                    "be an instance of a class derived from "
                    "bentoml.adapters.BaseInputAdapter "
                )

            # noinspection PyPep8Naming
            InputAdapter = args[0]
            input_adapter = InputAdapter(*args[1:], **kwargs)
            output_adapter = DefaultOutput()
        else:
            assert isinstance(input, BaseInputAdapter), (
                "API input parameter must be an instance of a class derived from "
                "bentoml.adapters.BaseInputAdapter"
            )
            input_adapter = input
            output_adapter = output or DefaultOutput()

        setattr(func, "_is_api", True)
        setattr(func, "_input_adapter", input_adapter)
        setattr(func, "_output_adapter", output_adapter)
        setattr(func, "_api_name", _api_name)
        setattr(func, "_api_route", _api_route)
        setattr(func, "_api_doc", _api_doc)
        setattr(func, "_mb_max_batch_size", mb_max_batch_size)
        setattr(func, "_mb_max_latency", mb_max_latency)
        setattr(func, "_batch", batch)

        return func

    return decorator


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
            'Invalid BentoService version: "{}", it can only consist'
            ' ALPHA / DIGIT / "-" / "." / "_", and must be less than'
            "128 characters".format(version_str)
        )

    if version_str.lower() == "latest":
        raise InvalidArgument('BentoService version can not be set to "latest"')



class Service:
    """
    BentoService is the base component for building prediction services using BentoML.

    BentoService provide an abstraction for describing model artifacts and environment
    dependencies required for a prediction service. And allows users to create inference
    APIs that defines the inferencing logic and how the underlying model can be served.
    Each BentoService can contain multiple models and serve multiple inference APIs.
    """

    # List of inference APIs that this BentoService provides
    _inference_apis: List[InferenceAPI] = []

    # Name of this BentoService. It is default the class name of this BentoService class
    _bento_service_name: str = None

    # For BentoService loaded from saved bundle, this will be set to the path of bundle.
    # When user install BentoService bundle as a PyPI package, this will be set to the
    # installed site-package location of current python environment
    _bento_service_bundle_path: str = None

    # List of artifacts required by this BentoService class, declared via the `@env`
    # decorator. This list is used for initializing an empty ArtifactCollection when
    # the BentoService class is instantiated
    _declared_artifacts: List[BentServiceArtifact] = []

    # An instance of ArtifactCollection, containing all required trained model artifacts
    _artifacts: ArtifactCollection = None

    #  A `BentoServiceEnv` instance specifying the required dependencies and all system
    #  environment setups
    _env = None

    # When loading BentoService from saved bundle, this will be set to the version of
    # the saved BentoService bundle
    _bento_service_bundle_version = None

    # See `ver_decorator` function above for more information
    _version_major = None
    _version_minor = None

    # See `web_static_content` function above for more
    _web_static_content = None

    def __init__(self):
        # When creating BentoService instance from a saved bundle, set version to the
        # version specified in the saved bundle
        self._bento_service_version = self.__class__._bento_service_bundle_version
        self._dev_server_bundle_path: tempfile.TemporaryDirectory = None
        self._dev_server_interrupt_event: multiprocessing.Event = None
        self._dev_server_process: subprocess.Process = None

        self._config_artifacts()
        self._config_inference_apis()
        self._config_environments()

    def _config_environments(self):
        self._env = self.__class__._env or BentoServiceEnv()

        for api in self._inference_apis:
            self._env.add_pip_packages(api.input_adapter.pip_dependencies)
            self._env.add_pip_packages(api.output_adapter.pip_dependencies)

        for artifact in self.artifacts.get_artifact_list():
            artifact.set_dependencies(self.env)

    def _config_inference_apis(self):
        self._inference_apis = []

        for _, function in inspect.getmembers(
            self.__class__,
            predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x),
        ):
            if hasattr(function, "_is_api"):
                api_name = getattr(function, "_api_name")
                route = getattr(function, "_api_route", None)
                api_doc = getattr(function, "_api_doc")
                input_adapter = getattr(function, "_input_adapter")
                output_adapter = getattr(function, "_output_adapter")
                mb_max_latency = getattr(function, "_mb_max_latency")
                mb_max_batch_size = getattr(function, "_mb_max_batch_size")
                batch = getattr(function, "_batch")

                # Bind api method call with self(BentoService instance)
                user_func = function.__get__(self)

                self._inference_apis.append(
                    InferenceAPI(
                        self,
                        api_name,
                        api_doc,
                        input_adapter=input_adapter,
                        user_func=user_func,
                        output_adapter=output_adapter,
                        mb_max_latency=mb_max_latency,
                        mb_max_batch_size=mb_max_batch_size,
                        batch=batch,
                        route=route,
                    )
                )

    def _config_artifacts(self):
        self._artifacts = ArtifactCollection.from_declared_artifact_list(
            self._declared_artifacts
        )

        if self._bento_service_bundle_path:
            # For pip installed BentoService, artifacts directory is located at
            # 'package_path/artifacts/', but for loading from bundle directory, it is
            # in 'path/{service_name}/artifacts/'
            if os.path.isdir(
                os.path.join(self._bento_service_bundle_path, ARTIFACTS_DIR_NAME)
            ):
                artifacts_path = os.path.join(
                    self._bento_service_bundle_path, ARTIFACTS_DIR_NAME
                )
            else:
                artifacts_path = os.path.join(
                    self._bento_service_bundle_path, self.name, ARTIFACTS_DIR_NAME
                )

            self.artifacts.load_all(artifacts_path)

    @property
    def inference_apis(self) -> List[InferenceAPI]:
        """Return a list of user defined API functions

        Returns:
            list(InferenceAPI): List of Inference API objects
        """
        return self._inference_apis

    def get_inference_api(self, api_name):
        """Find the inference API in this BentoService with a specific name.

        When the api_name is None, this returns the first Inference API found in the
        `self.inference_apis` list.

        :param api_name: the target Inference API's name
        :return:
        """
        if api_name:
            try:
                return next(
                    (api for api in self.inference_apis if api.name == api_name)
                )
            except StopIteration:
                raise NotFound(
                    "Can't find API '{}' in service '{}'".format(api_name, self.name)
                )
        elif len(self.inference_apis) > 0:
            return self.inference_apis[0]
        else:
            raise NotFound(f"Can't find any inference API in service '{self.name}'")

    @property
    def artifacts(self):
        """ Returns the ArtifactCollection instance specified with this BentoService
        class

        Returns:
            artifacts(ArtifactCollection): A dictionary of packed artifacts from the
            artifact name to the BentoServiceArtifact instance
        """
        return self._artifacts

    @property
    def env(self):
        return self._env

    @property
    def web_static_content(self):
        return self._web_static_content

    def get_web_static_content_path(self):
        if not self.web_static_content:
            return None
        if self._bento_service_bundle_path:
            return os.path.join(
                self._bento_service_bundle_path, self.name, 'web_static_content',
            )
        else:
            return os.path.join(os.getcwd(), self.web_static_content)

    @hybridmethod
    @property
    def name(self):
        """
        :return: BentoService name
        """
        return self.__class__.name  # pylint: disable=no-value-for-parameter

    @name.classmethod
    @property
    def name(cls):  # pylint: disable=no-self-argument,invalid-overridden-method
        """
        :return: BentoService name
        """
        if cls._bento_service_name is not None:
            if not cls._bento_service_name.isidentifier():
                raise InvalidArgument(
                    'BentoService#_bento_service_name must be valid python identifier'
                    'matching regex `(letter|"_")(letter|digit|"_")*`'
                )

            return cls._bento_service_name
        else:
            # Use python class name as service name
            return cls.__name__

    def set_version(self, version_str=None):
        """Set the version of this BentoService instance. Once the version is set
        explicitly via `set_version`, the `self.versioneer` method will no longer be
        invoked when saving this BentoService.
        """
        if version_str is None:
            version_str = self.versioneer()

        if self._version_major is not None and self._version_minor is not None:
            # BentoML uses semantic versioning for BentoService distribution
            # when user specified the MAJOR and MINOR version number along with
            # the BentoService class definition with '@ver' decorator.
            # The parameter version(or auto generated version) here will be used as
            # PATCH field in the final version:
            version_str = ".".join(
                [str(self._version_major), str(self._version_minor), version_str]
            )

        validate_version_str(version_str)

        if self.__class__._bento_service_bundle_version is not None:
            logger.warning(
                "Overriding loaded BentoService(%s) version:%s to %s",
                self.__class__._bento_service_bundle_path,
                self.__class__._bento_service_bundle_version,
                version_str,
            )
            self.__class__._bento_service_bundle_version = None

        if (
            self._bento_service_version is not None
            and self._bento_service_version != version_str
        ):
            logger.warning(
                "Resetting BentoService '%s' version from %s to %s",
                self.name,
                self._bento_service_version,
                version_str,
            )

        self._bento_service_version = version_str
        return self._bento_service_version

    def versioneer(self):
        """
        Function used to generate a new version string when saving a new BentoService
        bundle. User can also override this function to get a customized version format
        """
        datetime_string = datetime.now().strftime("%Y%m%d%H%M%S")
        random_hash = uuid.uuid4().hex[:6].upper()

        # Example output: '20191009135240_D246ED'
        return datetime_string + "_" + random_hash

    @property
    def version(self):
        """
        Return the version of this BentoService. If the version of this BentoService has
        not been set explicitly via `self.set_version`, a new version will be generated
        with the `self.versioneer` method. User can customize this version str either by
        setting the version with `self.set_version` before a `save` call, or override
        the `self.versioneer` method to customize the version str generator logic.

        For BentoService loaded from a saved bundle, this will simply return the version
        information found in the saved bundle.

        :return: BentoService version str
        """
        if self.__class__._bento_service_bundle_version is not None:
            return self.__class__._bento_service_bundle_version

        if self._bento_service_version is None:
            self.set_version(self.versioneer())

        return self._bento_service_version

    @property
    def tag(self):
        """
        Bento tag is simply putting its name and version together, separated by a colon
        `tag` is mostly used in Yatai model management related APIs and operations
        """
        return f"{self.name}:{self.version}"

    def save(self, yatai_url=None, version=None, labels=None):
        """
        Save and register this BentoService via BentoML's built-in model management
        system. BentoML by default keeps track of all the SavedBundle's files and
        metadata in local file system under the $BENTOML_HOME(~/bentoml) directory.
        Users can also configure BentoML to save their BentoService to a shared Database
        and cloud object storage such as AWS S3.

        :param yatai_url: optional - URL path to Yatai server
        :param version: optional - save with version override
        :param labels: optional - labels dictionary
        :return: saved_path: file path to where the BentoService is saved
        """
        from bentoml.yatai.client import get_yatai_client

        yc = get_yatai_client(yatai_url)

        return yc.repository.upload(self, version, labels)

    def save_to_dir(self, path, version=None):
        """Save this BentoService along with all its artifacts, source code and
        dependencies to target file path, assuming path exist and empty. If target path
        is not empty, this call may override existing files in the given path.

        :param path (str): Destination of where the bento service will be saved
        :param version: optional - save with version override
        """
        return save_to_dir(self, path, version)

    @hybridmethod
    def pack(self, name, *args, **kwargs):
        """
        BentoService#pack method is used for packing trained model instances with a
        BentoService instance and make it ready for BentoService#save.

        pack(name, *args, **kwargs):

        :param name: name of the declared model artifact
        :param args: args passing to the target model artifact to be packed
        :param kwargs: kwargs passing to the target model artifact to be packed
        :return: this BentoService instance
        """
        self.artifacts.get(name).pack(*args, **kwargs)
        return self

    @pack.classmethod
    def pack(cls, *args, **kwargs):  # pylint: disable=no-self-argument
        """
        **Deprecated**: Legacy `BentoService#pack` class method, no longer supported
        """
        raise BentoMLException(
            "BentoService#pack class method is deprecated, use instance method `pack` "
            "instead. e.g.: svc = MyBentoService(); svc.pack('model', model_object)"
        )

    def get_bento_service_metadata_pb(self):
        return SavedBundleConfig(self).get_bento_service_metadata_pb()

    pip_dependencies_map = None

    def start_dev_server(self, port=None, enable_ngrok=False, debug=False):
        if self._dev_server_process:
            logger.warning(
                "There is already a running dev server, "
                "please call `service.stop_dev_server()` first."
            )
            return
        try:
            self._dev_server_bundle_path = tempfile.TemporaryDirectory()
            self.save_to_dir(self._dev_server_bundle_path.name)

            def print_log(p):
                for line in p.stdout:
                    print(line.decode(), end='')

            def run(path, interrupt_event):
                my_env = os.environ.copy()
                # my_env["FLASK_ENV"] = "development"
                cmd = [sys.executable, "-m", "bentoml", "serve"]
                if port:
                    cmd += ['--port', f'{port}']
                if enable_ngrok:
                    cmd += ['--run-with-ngrok']
                if debug:
                    cmd += ['--debug']
                cmd += [path]
                p = subprocess.Popen(
                    cmd,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    env=my_env,
                )
                threading.Thread(target=print_log, args=(p,), daemon=True).start()
                interrupt_event.wait()
                p.terminate()

            self._dev_server_interrupt_event = multiprocessing.Event()
            self._dev_server_process = multiprocessing.Process(
                target=run,
                args=(
                    self._dev_server_bundle_path.name,
                    self._dev_server_interrupt_event,
                ),
                daemon=True,
            )
            self._dev_server_process.start()
            logger.info(
                f"======= starting dev server on port: {port if port else 5000} ======="
            )
        except Exception as e:  # pylint: disable=broad-except
            self.stop_dev_server(skip_log=True)
            raise e

    def stop_dev_server(self, skip_log=False):
        if self._dev_server_interrupt_event:
            self._dev_server_interrupt_event.set()
            self._dev_server_interrupt_event = None
        if self._dev_server_process:
            self._dev_server_process.join()
            assert not self._dev_server_process.is_alive()
            self._dev_server_process = None
            logger.info("Dev server has stopped.")
        elif not skip_log:
            logger.warning("No dev server is running.")
        if self._dev_server_bundle_path:
            self._dev_server_bundle_path.cleanup()
            self._dev_server_bundle_path = None

    def __del__(self):
        if hasattr(self, "_dev_server_interrupt_event"):  # __init__ may not be called
            self.stop_dev_server(skip_log=True)

    @inject
    def infer_pip_dependencies_map(
        self,
        bentoml_version: str = Provide[
            BentoMLContainer.bento_bundle_deployment_version
        ],
    ):
        if not self.pip_dependencies_map:
            self.pip_dependencies_map = {}
            bento_service_module = sys.modules[self.__class__.__module__]
            if hasattr(bento_service_module, "__file__"):
                bento_service_py_file_path = bento_service_module.__file__
                reqs, unknown_modules = seek_pip_packages(bento_service_py_file_path)
                self.pip_dependencies_map.update(reqs)
                for module_name in unknown_modules:
                    logger.warning(
                        "unknown package dependency for module: %s", module_name
                    )

            # Reset bentoml to configured deploy version - this is for users using
            # customized BentoML branch for development but use a different stable
            # version for deployment
            #
            # For example, a BentoService created with local dirty branch will fail
            # to deploy with docker due to the version can't be found on PyPI, but
            # get_bentoml_deploy_version gives the user the latest released PyPI
            # version that's closest to the `dirty` branch
            self.pip_dependencies_map['bentoml'] = bentoml_version

        return self.pip_dependencies_map
