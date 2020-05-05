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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import inspect
import logging
import uuid
from datetime import datetime
from abc import abstractmethod, ABCMeta

from bentoml import config
from bentoml.bundler import save_to_dir
from bentoml.bundler.config import SavedBundleConfig
from bentoml.service_env import BentoServiceEnv
from bentoml.utils import isidentifier
from bentoml.utils.hybridmethod import hybridmethod
from bentoml.marshal.utils import DataLoader
from bentoml.utils.trace import trace
from bentoml.exceptions import NotFound, InvalidArgument

ARTIFACTS_DIR_NAME = "artifacts"
ZIPKIN_API_URL = config("tracing").get("zipkin_api_url")

logger = logging.getLogger(__name__)


class BentoServiceAPI(object):
    """BentoServiceAPI defines abstraction for an API call that can be executed
    with BentoAPIServer and BentoCLI

    Args:
        service (BentoService): ref to service containing this API
        name (str): API name, by default this is the python function name
        handler (bentoml.handlers.BentoHandler): A BentoHandler class that transforms
            HTTP Request and/or CLI options into expected format for the API func
        func (function): API func contains the actual API callback, this is
            typically the 'predict' method on a model
    """

    def __init__(self, service, name, doc, handler, func):
        """
        :param service: ref to service containing this API
        :param name: API name
        :param handler: A BentoHandler that transforms HTTP Request and/or
            CLI options into parameters for API func
        :param func: API func contains the actual API callback, this is
            typically the 'predict' method on a model
        """
        self._service = service
        self._name = name
        self._doc = doc
        self._handler = handler
        self._func = func
        self._wrapped_func = None

    @property
    def service(self):
        return self._service

    @property
    def name(self):
        return self._name

    @property
    def doc(self):
        return self._doc

    @property
    def handler(self):
        return self._handler

    @property
    def func(self):
        if not self._wrapped_func:

            def _wrapped_func(*args, **kwargs):
                with trace(
                    ZIPKIN_API_URL,
                    service_name=self.__class__.__name__,
                    span_name="user defined api handler",
                ):
                    resp = self._func(*args, **kwargs)
                return resp

            self._wrapped_func = _wrapped_func
        return self._wrapped_func

    @property
    def request_schema(self):
        return self.handler.request_schema

    def handle_request(self, request):
        return self.handler.handle_request(request, self.func)

    def handle_batch_request(self, request):
        requests = DataLoader.split_requests(request.data)
        with trace(
            ZIPKIN_API_URL,
            service_name=self.__class__.__name__,
            span_name=f"call `{self._handler.__class__.__name__}`",
        ):
            responses = self.handler.handle_batch_request(requests, self.func)
        return DataLoader.merge_responses(responses)

    def handle_cli(self, args):
        return self.handler.handle_cli(args, self.func)

    def handle_aws_lambda_event(self, event):
        return self.handler.handle_aws_lambda_event(event, self.func)


class BentoServiceBase(object):
    """
    BentoServiceBase is an abstraction class that defines the interface for accesing a
    list of BentoServiceAPI for BentoAPIServer and BentoCLI to execute on
    """

    __metaclass__ = ABCMeta

    _service_apis = []

    @property
    @abstractmethod
    def name(self):
        """
        return BentoService name
        """

    @property
    @abstractmethod
    def version(self):
        """
        return BentoService version str
        """

    def _config_service_apis(self):
        self._service_apis = []
        for _, function in inspect.getmembers(
            self.__class__,
            predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x),
        ):
            if hasattr(function, "_is_api"):
                api_name = getattr(function, "_api_name")
                api_doc = getattr(function, "_api_doc")
                handler = getattr(function, "_handler")

                # Bind api method call with self(BentoService instance)
                func = function.__get__(self)

                self._service_apis.append(
                    BentoServiceAPI(self, api_name, api_doc, handler, func)
                )

    def get_service_apis(self):
        """Return a list of user defined API functions

        Returns:
            list(BentoServiceAPI): List of user defined API functions
        """
        return self._service_apis

    def get_service_api(self, api_name=None):
        if api_name:
            try:
                return next((api for api in self._service_apis if api.name == api_name))
            except StopIteration:
                raise NotFound(
                    "Can't find API '{}' in service '{}'".format(api_name, self.name)
                )
        elif len(self._service_apis):
            return self._service_apis[0]
        else:
            raise NotFound("Can't find default API for service '{}'".format(self.name))


def api_decorator(handler_cls, *args, **kwargs):
    """Decorator for adding api to a BentoService

    Args:
        handler_cls (bentoml.handlers.BentoHandler): The handler class for the API
            function.

        api_name (:obj:`str`, optional): API name to replace function name
        api_doc (:obj:`str`, optional): Docstring for API function
        **kwargs: Additional keyword arguments for handler class. Please reference
            to what arguments are available for the particular handler

    Raises:
        InvalidArgument: API name must contains only letters

    >>> from bentoml import BentoService, api
    >>> from bentoml.handlers import JsonHandler, DataframeHandler
    >>>
    >>> class FraudDetectionAndIdentityService(BentoService):
    >>>
    >>>     @api(JsonHandler)
    >>>     def fraud_detect(self, parsed_json):
    >>>         # do something
    >>>
    >>>     @api(DataframeHandler, input_json_orient='records')
    >>>     def identity(self, df):
    >>>         # do something

    """

    DEFAULT_API_DOC = "BentoService API"

    from bentoml.handlers.base_handlers import BentoHandler

    if not (inspect.isclass(handler_cls) and issubclass(handler_cls, BentoHandler)):
        raise InvalidArgument(
            "BentoService @api decorator first parameter must "
            "be class derived from bentoml.handlers.BentoHandler"
        )

    def decorator(func):
        api_name = kwargs.pop("api_name", func.__name__)
        api_doc = kwargs.pop("api_doc", func.__doc__ or DEFAULT_API_DOC).strip()

        handler = handler_cls(
            *args, **kwargs
        )  # create handler instance and attach to api method

        setattr(func, "_is_api", True)
        setattr(func, "_handler", handler)
        if not isidentifier(api_name):
            raise InvalidArgument(
                "Invalid API name: '{}', a valid identifier must contains only letters,"
                " numbers, underscores and not starting with a number.".format(api_name)
            )
        setattr(func, "_api_name", api_name)
        setattr(func, "_api_doc", api_doc)

        return func

    return decorator


def artifacts_decorator(artifacts):
    """Define artifacts required to be bundled with a BentoService

    Args:
        artifacts (list(bentoml.artifact.BentoServiceArtifact)): A list of desired
            artifacts required by this BentoService
    """
    from bentoml.artifact import BentoServiceArtifact

    def decorator(bento_service_cls):
        artifact_names = set()
        for artifact in artifacts:
            if not isinstance(artifact, BentoServiceArtifact):
                raise InvalidArgument(
                    "BentoService @artifacts decorator only accept list of type "
                    "BentoServiceArtifact, instead got type: '%s'" % type(artifact)
                )

            if artifact.name in artifact_names:
                raise InvalidArgument(
                    "Duplicated artifact name `%s` detected. Each artifact within one"
                    "BentoService must have an unique name" % artifact.name
                )

            artifact_names.add(artifact.name)

        bento_service_cls._artifacts = artifacts
        return bento_service_cls

    return decorator


def env_decorator(
    pip_dependencies=None,
    auto_pip_dependencies=False,
    requirements_txt_file=None,
    conda_channels=None,
    conda_dependencies=None,
    setup_sh=None,
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
        conda_channels: extra conda channels to be used
        conda_dependencies: list of conda dependencies required
        setup_sh: user defined setup bash script, it is executed in docker build time
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
    from bentoml.yatai import get_yatai_service

    if base_path:
        yatai_service = get_yatai_service(repo_base_url=base_path)
        yatai_client = YataiClient(yatai_service)
    else:
        yatai_client = YataiClient()

    return yatai_client.repository.upload(bento_service, version)


class BentoService(BentoServiceBase):
    """
    BentoService is the base component for building prediction services using BentoML.

    BentoService provide an abstraction for describing model artifacts and environment
    dependencies required for a prediction service. And allows users to write custom
    prediction API handling logic via BentoService API callback function.

    Each BentoService can contain multiple models via the BentoML Artifact class, and
    can define multiple APIs for accessing this service. Each API should specify a type
    of Handler, which defines the expected input data format for this API.

    >>>  from bentoml import BentoService, env, api, artifacts, ver
    >>>  from bentoml.handlers import DataframeHandler
    >>>  from bentoml.artifact import SklearnModelArtifact
    >>>
    >>>  @ver(major=1, minor=4)
    >>>  @artifacts([SklearnModelArtifact('clf')])
    >>>  @env(pip_dependencies=["scikit-learn"])
    >>>  class MyMLService(BentoService):
    >>>
    >>>     @api(DataframeHandler)
    >>>     def predict(self, df):
    >>>         return self.artifacts.clf.predict(df)
    >>>
    >>>  bento_service = MyMLService()
    >>>  bento_service.pack('clf', trained_classifier_model)
    >>>  bento_service.save_to_dir('/bentoml_bundles')
    """

    # User may use @name to override this if they don't want the generated model
    # to have the same name as their Python model class name
    _bento_service_name = None

    # For BentoService loaded from saved bundle, this will be set to the path of bundle.
    # When user install BentoService bundle as a PyPI package, this will be set to the
    # installed site-package location of current python environment
    _bento_service_bundle_path = None

    # list of artifacts required by this BentoService
    _artifacts = []

    # Describe the desired environment for this BentoService using
    # `bentoml.service_env.BentoServiceEnv`
    _env = None

    # When loading BentoService from saved bundle, this will be set to the version of
    # the saved BentoService bundle
    _bento_service_bundle_version = None

    # See `ver_decorator` function above for more information
    _version_major = None
    _version_minor = None

    def __init__(self):
        from bentoml.artifact import ArtifactCollection

        self._bento_service_version = self.__class__._bento_service_bundle_version
        self._packed_artifacts = ArtifactCollection()

        if self._bento_service_bundle_path:
            # load artifacts from saved BentoService bundle
            self._load_artifacts(self._bento_service_bundle_path)

        self._config_service_apis()
        self._init_env()

    def _init_env(self):
        self._env = self.__class__._env or BentoServiceEnv(self.name)

        for api in self._service_apis:
            self._env._add_pip_dependencies(api.handler.pip_dependencies)

        for artifact in self._artifacts:
            self._env._add_pip_dependencies(artifact.pip_dependencies)

    @property
    def artifacts(self):
        """
        :return: List of model artifacts
        """
        return self._packed_artifacts

    @property
    def env(self):
        return self._env

    @hybridmethod
    @property
    def name(self):
        return self.__class__.name()  # pylint: disable=no-value-for-parameter

    @name.classmethod
    def name(cls):  # pylint: disable=no-self-argument,invalid-overridden-method
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
        """Manually override the version of this BentoService instance
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
        if name in self.artifacts:
            logger.warning(
                "BentoService '%s' #pack overriding existing artifact '%s'",
                self.name,
                name,
            )
            del self.artifacts[name]

        artifact = next(
            artifact for artifact in self._artifacts if artifact.name == name
        )
        packed_artifact = artifact.pack(*args, **kwargs)
        self._packed_artifacts.add(packed_artifact)
        return self

    @pack.classmethod
    def pack(cls, *args, **kwargs):  # pylint: disable=no-self-argument
        """
        **Deprecated**: Legacy `BentoService#pack` class method, which can be used to
        initialize a BentoService instance along with trained model artifacts. This will
        be deprecated soon:

        :param args: args passing to the BentoService class
        :param kwargs: kwargs passing to the BentoService class and (artifact_name,
            args) pair for creating declared model artifacts
        :return: a new BentoService instance
        """
        logger.warning(
            "BentoService#pack class method is deprecated, use instance method `pack` "
            "instead. e.g.: svc = MyBentoService(); svc.pack('model', model_object)"
        )
        from bentoml.artifact import ArtifactCollection

        if args and isinstance(args[0], ArtifactCollection):
            bento_svc = cls(*args[1:], **kwargs)  # pylint: disable=not-callable
            bento_svc._packed_artifacts = args[0]
            return bento_svc

        packed_artifacts = []
        for artifact in cls._artifacts:
            if artifact.name in kwargs:
                artifact_args = kwargs.pop(artifact.name)
                packed_artifacts.append(artifact.pack(artifact_args))

        bento_svc = cls(*args, **kwargs)  # pylint: disable=not-callable
        for packed_artifact in packed_artifacts:
            bento_svc.artifacts.add(packed_artifact)

        return bento_svc

    def _load_artifacts(self, path):
        # For pip installed BentoService, artifacts directory is located at
        # 'package_path/artifacts/', but for loading from bundle directory, it is
        # in 'path/{service_name}/artifacts/'
        if not os.path.isdir(os.path.join(path, ARTIFACTS_DIR_NAME)):
            artifacts_path = os.path.join(path, self.name, ARTIFACTS_DIR_NAME)
        else:
            artifacts_path = os.path.join(path, ARTIFACTS_DIR_NAME)

        for artifact in self._artifacts:
            packed_artifact = artifact.load(artifacts_path)
            self._packed_artifacts.add(packed_artifact)

    def get_bento_service_metadata_pb(self):
        return SavedBundleConfig(self).get_bento_service_metadata_pb()
