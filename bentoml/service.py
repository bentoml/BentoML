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

import os
import re
import inspect
import logging
import uuid
from datetime import datetime
from typing import List

import flask
from werkzeug.utils import cached_property

from bentoml import config
from bentoml.saved_bundle import save_to_dir
from bentoml.saved_bundle.config import SavedBundleConfig
from bentoml.service_env import BentoServiceEnv
from bentoml.utils import isidentifier
from bentoml.utils.hybridmethod import hybridmethod
from bentoml.exceptions import NotFound, InvalidArgument, BentoMLException
from bentoml.server import trace
from bentoml.artifact import BentoServiceArtifact, ArtifactCollection
from bentoml.adapters import BaseInputAdapter, BaseOutputAdapter


ARTIFACTS_DIR_NAME = "artifacts"
DEFAULT_MAX_LATENCY = config("marshal_server").getint("default_max_latency")
DEFAULT_MAX_BATCH_SIZE = config("marshal_server").getint("default_max_batch_size")
BENTOML_RESERVED_API_NAMES = [
    "index",
    "swagger",
    "docs",
    "healthz",
    "metrics",
    "feedback",
]
logger = logging.getLogger(__name__)


class InferenceAPI(object):
    """
    InferenceAPI defines an inference call to the underlying model, including its input
    and output adapter, the user-defined API callback function, and configurations for
    working with the BentoML adaptive micro-batching mechanism
    """

    def __init__(
        self, service, name, doc, handler, func, mb_max_latency, mb_max_batch_size
    ):
        """
        :param service: ref to service containing this API
        :param name: API name
        :param doc: the user facing document of this inference API, default to the
            docstring of the inference API function
        :param handler: A InputAdapter that transforms HTTP Request and/or
            CLI options into parameters for API func
        :param func: the user-defined API callback function, this is
            typically the 'predict' method on a model
        :param mb_max_latency: The latency goal of this inference API in milliseconds.
            Default: 10000.
        :param mb_max_batch_size: The maximum size of requests batch accepted by this
            inference API. This parameter governs the throughput/latency trade off, and
            avoids having large batches that exceed some resource constraint (e.g. GPU
            memory to hold the entire batch's data). Default: 1000.

        """
        self._service = service
        self._name = name
        self._handler = handler
        self._func = func
        self._wrapped_func = None
        self.mb_max_latency = mb_max_latency
        self.mb_max_batch_size = mb_max_batch_size

        if doc is None:
            # generate a default doc string for this inference API
            doc = (
                f"BentoService inference API '{self.name}', input: "
                f"'{type(handler).__name__}', output: "
                f"'{type(handler.output_adapter).__name__}'"
            )
        self._doc = doc

    @property
    def service(self):
        """
        :return: a reference to the BentoService serving this inference API
        """
        return self._service

    @property
    def name(self):
        """
        :return: the name of this inference API
        """
        return self._name

    @property
    def doc(self):
        """
        :return: user facing documentation of this inference API
        """
        return self._doc

    @property
    def handler(self):
        """
        handler is a deprecated concept, we are moving it to the new input/output
        adapter interface. Currently this `handler` property returns the input adapter
        of this inference API
        """
        return self._handler

    @property
    def output_adapter(self):
        """
        :return: the output adapter of this inference API
        """
        return self.handler.output_adapter

    @cached_property
    def func(self):
        """
        :return: user-defined inference API callback function
        """

        def func_with_tracing(*args, **kwargs):
            with trace(
                service_name=self.__class__.__name__,
                span_name="user defined inference api callback function",
            ):
                return self._func(*args, **kwargs)

        return func_with_tracing

    @property
    def request_schema(self):
        """
        :return: the HTTP API request schema in OpenAPI/Swagger format
        """
        schema = self.handler.request_schema
        if schema.get('application/json'):
            schema.get('application/json')['example'] = self.handler._http_input_example
        return schema

    def handle_request(self, request: flask.Request):
        return self.handler.handle_request(request, self.func)

    def handle_batch_request(self, request: flask.Request):
        from bentoml.marshal.utils import DataLoader

        requests = DataLoader.split_requests(request.get_data())

        with trace(
            service_name=self.__class__.__name__,
            span_name=f"call `{self.handler.__class__.__name__}`",
        ):
            responses = self.handler.handle_batch_request(requests, self.func)

        return DataLoader.merge_responses(responses)

    def handle_cli(self, args):
        return self.handler.handle_cli(args, self.func)

    def handle_aws_lambda_event(self, event):
        return self.handler.handle_aws_lambda_event(event, self.func)


def validate_inference_api_name(api_name):
    if not isidentifier(api_name):
        raise InvalidArgument(
            "Invalid API name: '{}', a valid identifier must contains only letters,"
            " numbers, underscores and not starting with a number.".format(api_name)
        )

    if api_name in BENTOML_RESERVED_API_NAMES:
        raise InvalidArgument(
            "Reserved API name: '{}' is reserved for infra endpoints".format(api_name)
        )


def api_decorator(
    *args,
    input: BaseInputAdapter = None,
    output: BaseOutputAdapter = None,
    api_name=None,
    api_doc=None,
    mb_max_batch_size=DEFAULT_MAX_BATCH_SIZE,
    mb_max_latency=DEFAULT_MAX_LATENCY,
    **kwargs,
):  # pylint: disable=redefined-builtin
    """
    A decorator exposed as `bentoml.api` for defining Inference API in a BentoService
    class.

    :param input: InputAdapter instance of the inference API
    :param output: OutputAdapter instance of the inference API
    :param api_name: API name, default to the user-defined callback function's function
        name
    :param api_doc: user-facing documentation of the inference API. default to the
        user-defined callback function's docstring
    :param mb_max_batch_size: The maximum size of requests batch accepted by this
        inference API. This parameter governs the throughput/latency trade off, and
        avoids having large batches that exceed some resource constraint (e.g. GPU
        memory to hold the entire batch's data). Default: 1000.
    :param mb_max_latency: The latency goal of this inference API in milliseconds.
        Default: 10000.


    Example usage:

    >>> from bentoml import BentoService, api
    >>> from bentoml.adapters import JsonInput, DataframeInput
    >>>
    >>> class FraudDetectionAndIdentityService(BentoService):
    >>>
    >>>     @api(input=JsonInput())
    >>>     def fraud_detect(self, json_list):
    >>>         # user-defined callback function that process inference requests
    >>>
    >>>     @api(input=DataframeInput(input_json_orient='records'))
    >>>     def identity(self, df):
    >>>         # user-defined callback function that process inference requests


    Deprecated syntax before version 0.8.0:

    >>> from bentoml import BentoService, api
    >>> from bentoml.handlers import JsonHandler, DataframeHandler  # deprecated
    >>>
    >>> class FraudDetectionAndIdentityService(BentoService):
    >>>
    >>>     @api(JsonHandler)  # deprecated
    >>>     def fraud_detect(self, parsed_json_list):
    >>>         # do something
    >>>
    >>>     @api(DataframeHandler, input_json_orient='records')  # deprecated
    >>>     def identity(self, df):
    >>>         # do something

    """

    def decorator(func):
        _api_name = func.__name__ if api_name is None else api_name
        validate_inference_api_name(_api_name)

        if input is None:
            # support bentoml<=0.7
            if not args or not (
                inspect.isclass(args[0]) and issubclass(args[0], BaseInputAdapter)
            ):
                raise InvalidArgument(
                    "BentoService @api decorator first parameter must "
                    "be class derived from bentoml.adapters.BaseInputAdapter"
                )

            handler = args[0](*args[1:], output_adapter=output, **kwargs)
        else:
            assert isinstance(input, BaseInputAdapter), (
                "API input parameter must be an instance of any classes inherited from "
                "bentoml.adapters.BaseInputAdapter"
            )
            handler = input
            handler._output_adapter = output

        setattr(func, "_is_api", True)
        setattr(func, "_handler", handler)
        setattr(func, "_api_name", _api_name)
        setattr(func, "_api_doc", api_doc)
        setattr(func, "_mb_max_batch_size", mb_max_batch_size)
        setattr(func, "_mb_max_latency", mb_max_latency)

        return func

    return decorator


def web_static_content_decorator(web_static_content):
    """Define web UI static files required to be bundled with a BentoService

    Args:
        web_static_content: path to directory containg index.html and static dir

    >>>  @web_static_content('./ui/')
    >>>  class MyMLService(BentoService):
    >>>     pass
    """

    def decorator(bento_service_cls):
        bento_service_cls._web_static_content = web_static_content
        return bento_service_cls

    return decorator


def artifacts_decorator(artifacts: List[BentoServiceArtifact]):
    """Define artifacts required to be bundled with a BentoService

    Args:
        artifacts (list(bentoml.artifact.BentoServiceArtifact)): A list of desired
            artifacts required by this BentoService
    """

    def decorator(bento_service_cls):
        artifact_names = set()
        for artifact in artifacts:
            if not isinstance(artifact, BentoServiceArtifact):
                raise InvalidArgument(
                    "BentoService @artifacts decorator only accept list of "
                    "BentoServiceArtifact instances, instead got type: '%s'"
                    % type(artifact)
                )

            if artifact.name in artifact_names:
                raise InvalidArgument(
                    "Duplicated artifact name `%s` detected. Each artifact within one"
                    "BentoService must have an unique name" % artifact.name
                )

            artifact_names.add(artifact.name)

        bento_service_cls._declared_artifacts = artifacts
        return bento_service_cls

    return decorator


def env_decorator(
    pip_dependencies: List[str] = None,
    auto_pip_dependencies: bool = False,
    requirements_txt_file: str = None,
    conda_channels: List[str] = None,
    conda_dependencies: List[str] = None,
    setup_sh: str = None,
    docker_base_image: str = None,
):
    """Define environment and dependencies required for the BentoService being created

    Args:
        pip_dependencies: list of pip_dependencies required, specified by package name
            or with specified version `{package_name}=={package_version}`
        auto_pip_dependencies: (Beta) whether to automatically find all the required
            pip dependencies and pin their version
        requirements_txt_file: pip dependencies in the form of a requirements.txt file,
            this can be a relative path to the requirements.txt file or the content
            of the file
        conda_channels: list of extra conda channels to be used
        conda_dependencies: list of conda dependencies required
        setup_sh: user defined setup bash script, it is executed in docker build time
        docker_base_image: used for customizing the docker container image built with
            BentoML saved bundle. Base image must either have both `bash` and `conda`
            installed; or have `bash`, `pip`, `python` installed, in which case the user
            is required to ensure the python version matches the BentoService bundle
    """

    def decorator(bento_service_cls):
        bento_service_cls._env = BentoServiceEnv(
            bento_service_name=bento_service_cls.name(),
            pip_dependencies=pip_dependencies,
            auto_pip_dependencies=auto_pip_dependencies,
            requirements_txt_file=requirements_txt_file,
            conda_channels=conda_channels,
            conda_dependencies=conda_dependencies,
            setup_sh=setup_sh,
            docker_base_image=docker_base_image,
        )
        return bento_service_cls

    return decorator


def ver_decorator(major, minor):
    """Decorator for specifying the version of a custom BentoService.

    Args:
        major (int): Major version number for Bento Service
        minor (int): Minor version number for Bento Service

    BentoML uses semantic versioning for BentoService distribution:

    * MAJOR is incremented when you make breaking API changes

    * MINOR is incremented when you add new functionality without breaking the
      existing API or functionality

    * PATCH is incremented when you make backwards-compatible bug fixes

    'Patch' is provided(or auto generated) when calling BentoService#save,
    while 'Major' and 'Minor' can be defined with '@ver' decorator

    >>>  @ver(major=1, minor=4)
    >>>  @artifacts([PickleArtifact('model')])
    >>>  class MyMLService(BentoService):
    >>>     pass
    >>>
    >>>  svc = MyMLService()
    >>>  svc.pack("model", trained_classifier)
    >>>  svc.set_version("2019-08.iteration20")
    >>>  svc.save()
    >>>  # The final produced BentoService bundle will have version:
    >>>  # "1.4.2019-08.iteration20"
    """

    def decorator(bento_service_cls):
        bento_service_cls._version_major = major
        bento_service_cls._version_minor = minor
        return bento_service_cls

    return decorator


def _validate_version_str(version_str):
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
            "128 characthers".format(version_str)
        )

    if version_str.lower() == "latest":
        raise InvalidArgument('BentoService version can not be set to "latest"')


def save(bento_service, base_path=None, version=None):
    """
    Save and register the given BentoService via BentoML's built-in model management
    system. BentoML by default keeps track of all the SavedBundle's files and metadata
    in local file system under the $BENTOML_HOME(~/bentoml) directory. Users can also
    configure BentoML to save their BentoService to a shared Database and cloud object
    storage such as AWS S3.

    :param bento_service: target BentoService instance to be saved
    :param base_path: optional - override repository base path
    :param version: optional - save with version override
    :return: saved_path: file path to where the BentoService is saved
    """

    from bentoml.yatai.client import YataiClient
    from bentoml.yatai.yatai_service import get_yatai_service

    if base_path:
        yatai_service = get_yatai_service(repo_base_url=base_path)
        yatai_client = YataiClient(yatai_service)
    else:
        yatai_client = YataiClient()

    return yatai_client.repository.upload(bento_service, version)


class BentoService:
    """
    BentoService is the base component for building prediction services using BentoML.

    BentoService provide an abstraction for describing model artifacts and environment
    dependencies required for a prediction service. And allows users to create inference
    APIs that defines the inferencing logic and how the underlying model can be served.
    Each BentoService can contain multiple models and serve multiple inference APIs.

    Usage example:

    >>>  from bentoml import BentoService, env, api, artifacts
    >>>  from bentoml.adapters import DataframeInput
    >>>  from bentoml.artifact import SklearnModelArtifact
    >>>
    >>>  @artifacts([SklearnModelArtifact('clf')])
    >>>  @env(pip_dependencies=["scikit-learn"])
    >>>  class MyMLService(BentoService):
    >>>
    >>>     @api(input=DataframeInput())
    >>>     def predict(self, df):
    >>>         return self.artifacts.clf.predict(df)
    >>>
    >>>  if __name__ == "__main__":
    >>>     bento_service = MyMLService()
    >>>     bento_service.pack('clf', trained_classifier_model)
    >>>     bento_service.save_to_dir('/bentoml_bundles')
    """

    # List of inference APIs that this BentoService provides
    _inference_apis = []

    # Name of this BentoService. It is default the class name of this BentoService class
    _bento_service_name = None

    # For BentoService loaded from saved bundle, this will be set to the path of bundle.
    # When user install BentoService bundle as a PyPI package, this will be set to the
    # installed site-package location of current python environment
    _bento_service_bundle_path = None

    # List of artifacts required by this BentoService class, declared via the `@env`
    # decorator. This list is used for initializing an empty ArtifactCollection when
    # the BentoService class is instantiated
    _declared_artifacts: List[BentoServiceArtifact] = []

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

        self._config_artifacts()
        self._config_inference_apis()
        self._config_environments()

    def _config_environments(self):
        self._env = self.__class__._env or BentoServiceEnv(self.name)

        for api in self._inference_apis:
            self._env.add_pip_dependencies_if_missing(api.handler.pip_dependencies)
            self._env.add_pip_dependencies_if_missing(
                api.output_adapter.pip_dependencies
            )

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
                api_doc = getattr(function, "_api_doc")
                handler = getattr(function, "_handler")
                mb_max_latency = getattr(function, "_mb_max_latency")
                mb_max_batch_size = getattr(function, "_mb_max_batch_size")

                # Bind api method call with self(BentoService instance)
                func = function.__get__(self)

                self._inference_apis.append(
                    InferenceAPI(
                        self,
                        api_name,
                        api_doc,
                        handler=handler,
                        func=func,
                        mb_max_latency=mb_max_latency,
                        mb_max_batch_size=mb_max_batch_size,
                    )
                )

    def _config_artifacts(self):
        self._artifacts = ArtifactCollection.from_artifact_list(
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
    def inference_apis(self):
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
        return self.__class__.name()  # pylint: disable=no-value-for-parameter

    @name.classmethod
    def name(cls):  # pylint: disable=no-self-argument,invalid-overridden-method
        """
        :return: BentoService name
        """
        if cls._bento_service_name is not None:
            if not isidentifier(cls._bento_service_name):
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

        _validate_version_str(version_str)

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

    def save(self, base_path=None, version=None):
        """
        Save and register this BentoService via BentoML's built-in model management
        system. BentoML by default keeps track of all the SavedBundle's files and
        metadata in local file system under the $BENTOML_HOME(~/bentoml) directory.
        Users can also configure BentoML to save their BentoService to a shared Database
        and cloud object storage such as AWS S3.

        :param base_path: optional - override repository base path
        :param version: optional - save with version override
        :return: saved_path: file path to where the BentoService is saved
        """
        return save(self, base_path, version)

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
