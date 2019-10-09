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
import sys
import inspect
import logging
import uuid
from datetime import datetime

from six import add_metaclass
from abc import abstractmethod, ABCMeta

from bentoml.archive import save_to_dir
from bentoml.exceptions import BentoMLException
from bentoml.service_env import BentoServiceEnv
from bentoml.artifact import ArtifactCollection
from bentoml.utils import isidentifier
from bentoml.proto.repository_pb2 import BentoServiceMetadata

logger = logging.getLogger(__name__)


def _get_func_attr(func, attribute_name):
    if sys.version_info.major < 3 and inspect.ismethod(func):
        func = func.__func__
    return getattr(func, attribute_name)


def _set_func_attr(func, attribute_name, value):
    if sys.version_info.major < 3 and inspect.ismethod(func):
        func = func.__func__
    return setattr(func, attribute_name, value)


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
        return self._func

    @property
    def request_schema(self):
        return self.handler.request_schema

    def handle_request(self, request):
        return self.handler.handle_request(request, self.func)

    def handle_cli(self, args):
        return self.handler.handle_cli(args, self.func)

    def handle_aws_lambda_event(self, event):
        return self.handler.handle_aws_lambda_event(event, self.func)

    def handle_clipper_strings(self, inputs):
        return self.handler.handle_clipper_strings(inputs, self.func)

    def handle_clipper_bytes(self, inputs):
        return self.handler.handle_clipper_bytes(inputs, self.func)

    def handle_clipper_ints(self, inputs):
        return self.handler.handle_clipper_ints(inputs, self.func)

    def handle_clipper_doubles(self, inputs):
        return self.handler.handle_clipper_doubles(inputs, self.func)

    def handle_clipper_floats(self, inputs):
        return self.handler.handle_clipper_floats(inputs, self.func)


@add_metaclass(ABCMeta)
class BentoServiceBase(object):
    """
    BentoServiceBase is the base abstraction that exposes a list of APIs
    for BentoAPIServer and BentoCLI to execute
    """

    _service_apis = []

    @property
    @abstractmethod
    def name(self):
        """
        return bento service name
        """

    @property
    @abstractmethod
    def version(self):
        """
        return bento service version str
        """

    def _config_service_apis(self):
        self._service_apis = []
        for _, function in inspect.getmembers(
            self.__class__,
            predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x),
        ):
            if hasattr(function, "_is_api"):
                api_name = _get_func_attr(function, "_api_name")
                api_doc = _get_func_attr(function, "_api_doc")
                handler = _get_func_attr(function, "_handler")

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
                raise ValueError(
                    "Can't find API '{}' in service '{}'".format(api_name, self.name)
                )
        elif len(self._service_apis):
            return self._service_apis[0]
        else:
            raise ValueError(
                "Can't find default API for service '{}'".format(self.name)
            )


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
        ValueError: API name must contains only letters

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

    DEFAULT_API_DOC = "BentoML generated API endpoint"

    def decorator(func):
        api_name = kwargs.pop("api_name", func.__name__)
        api_doc = kwargs.pop("api_doc", func.__doc__ or DEFAULT_API_DOC).strip()

        handler = handler_cls(
            *args, **kwargs
        )  # create handler instance and attach to api method

        _set_func_attr(func, "_is_api", True)
        _set_func_attr(func, "_handler", handler)
        if not isidentifier(api_name):
            raise ValueError(
                "Invalid API name: '{}', a valid identifier must contains only letters,"
                " numbers, underscores and not starting with a number.".format(api_name)
            )
        _set_func_attr(func, "_api_name", api_name)
        _set_func_attr(func, "_api_doc", api_doc)

        return func

    return decorator


def artifacts_decorator(artifact_specs):
    """Define artifact spec for BentoService

    Args:
        artifact_specs (list(bentoml.artifact.BentoServiceArtifact)): A list of desired
            artifacts for initializing this BentoService
        for initializing this BentoService being decorated
    """

    def decorator(bento_service_cls):
        bento_service_cls._artifacts_spec = artifact_specs
        return bento_service_cls

    return decorator


def env_decorator(**kwargs):
    """Define environment spec for BentoService

    Args:
        setup_sh (str): User defined shell script to run before running BentoService.
            It could be local file path or the shell script content.
        requirements_text (str): User defined requirement text to install before
            running BentoService.
        pip_dependencies (str or list(str)): User defined python modules to install.
        conda_channels (list(str)): User defined conda channels
        conda_dependencies (list(str)): Defined dependencies to be installed with
            conda environment.
        conda_pip_dependencies (list(str)): Additional pip modules to be install
            with conda

    """

    def decorator(bento_service_cls):
        bento_service_cls._env = BentoServiceEnv.from_dict(kwargs)
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
    >>>  svc = MyMLService.pack(model="my ML model object")
    >>>  svc.set_version("2019-08.iteration20")
    >>>  svc.save('/path_to_archive')
    >>>  # The final produced BentoArchive version will be "1.4.2019-08.iteration20"
    """

    def decorator(bento_service_cls):
        bento_service_cls._version_major = major
        bento_service_cls._version_minor = minor
        return bento_service_cls

    return decorator


def _validate_version_str(version_str):
    """
    Validate that version str format:
    * Consist of only ALPHA / DIGIT / "-" / "." / "_"
    * Length between 1-128
    """
    regex = r"[A-Za-z0-9_.-]{1,128}\Z"
    if re.match(regex, version_str) is None:
        raise ValueError(
            'Invalid BentoService version: "{}", it can only consist'
            ' ALPHA / DIGIT / "-" / "." / "_", and must be less than'
            "128 characthers".format(version_str)
        )


class BentoService(BentoServiceBase):
    """BentoService packs a list of artifacts and exposes service APIs
    for BentoAPIServer and BentoCLI to execute. By subclassing BentoService,
    users can customize the artifacts and environments required for
    a ML service.

    >>>  from bentoml import BentoService, env, api, artifacts, ver
    >>>  from bentoml.handlers import DataframeHandler
    >>>
    >>>  @ver(major=1, minor=4)
    >>>  @artifacts([PickleArtifact('clf')])
    >>>  @env(conda_dependencies: [ 'scikit-learn' ])
    >>>  class MyMLService(BentoService):
    >>>
    >>>     @api(DataframeHandler)
    >>>     def predict(self, df):
    >>>         return self.artifacts.clf.predict(df)
    >>>
    >>>  bento_service = MyMLService.pack(clf=my_trained_clf_object)
    >>>  bentoml.save_to_dir(bento_service, './export')
    """

    # User may use @name to override this if they don't want the generated model
    # to have the same name as their Python model class name
    _bento_service_name = None

    # For BentoService loaded from achive directory, this will be set to archive path
    # when user install exported bento model as a PyPI package, this will be set to
    # the python installed site-package location
    _bento_archive_path = None

    # list of artifact spec describing required artifacts for this BentoService
    _artifacts_spec = []
    _artifacts = None

    # Describe the desired environment for this BentoService using
    # `bentoml.service_env.BentoServiceEnv`
    _env = {}

    # This can only be set by BentoML library when loading from archive
    _bento_service_version = None

    # See `ver_decorator` function above for more information
    _version_major = None
    _version_minor = None

    def __init__(self, artifacts=None, env=None):
        self._init_artifacts(artifacts)
        self._config_service_apis()
        self._init_env(env)
        self.name = self.__class__.name()

    def _init_artifacts(self, artifacts):
        if len(self._artifacts_spec) > 0:
            if artifacts is None:
                if self._bento_archive_path:
                    artifacts = ArtifactCollection.load(
                        self._bento_archive_path, self.__class__._artifacts_spec
                    )
                else:
                    raise BentoMLException(
                        "Must provide required artifacts before instantiating a "
                        "custom BentoService class"
                    )

            if isinstance(artifacts, ArtifactCollection):
                self._artifacts = artifacts
            else:
                self._artifacts = ArtifactCollection()
                for artifact in artifacts:
                    self._artifacts[artifact.name] = artifact
        else:
            self._artifacts = ArtifactCollection()

    def _init_env(self, env=None):
        if env is None:
            # By default use BentoServiceEnv defined on class via @env decorator
            env = self.__class__._env

        if isinstance(env, dict):
            self._env = BentoServiceEnv.from_dict(env)
        else:
            self._env = env

        for api in self._service_apis:
            self._env.add_handler_dependencies(api.handler.pip_dependencies)

    @property
    def artifacts(self):
        return self._artifacts

    @property
    def env(self):
        return self._env

    @classmethod
    def name(cls):  # pylint:disable=method-hidden
        if cls._bento_service_name is not None:
            if not isidentifier(cls._bento_service_name):
                raise ValueError(
                    'BentoService#_bento_service_name must be valid python identifier'
                    'matching regex `(letter|"_")(letter|digit|"_")*`'
                )

            return cls._bento_service_name
        else:
            # Use python class name as service name
            return cls.__name__

    def set_version(self, version_str=None):
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
        self._bento_service_version = version_str
        return self._bento_service_version

    def versioneer(self):
        """
        Function used to generate a new version string when saving an archive of this
        user defined BentoService. User can also override this function to get a
        customized version format
        """
        datetime_string = datetime.now().strftime("%Y%m%d%H%M%S")
        random_hash = uuid.uuid4().hex[:6].upper()

        # Example output: '20191009135240_D246ED'
        return datetime_string + "_" + random_hash

    @property
    def version(self):
        if self._bento_service_version is None:
            self.set_version(self.versioneer())

        return self._bento_service_version

    def save(self, base_path=None, version=None):
        from bentoml.yatai import python_api

        return python_api.upload_bento_service(self, base_path, version)

    def save_to_dir(self, path, version=None):
        return save_to_dir(self, path, version)

    @classmethod
    def pack(cls, *args, **kwargs):
        if args and isinstance(args[0], ArtifactCollection):
            return cls(args[0])

        artifacts = ArtifactCollection()

        for artifact_spec in cls._artifacts_spec:
            if artifact_spec.name in kwargs:
                artifact_instance = artifact_spec.pack(kwargs[artifact_spec.name])
                artifacts.add(artifact_instance)

        return cls(artifacts)

    @classmethod
    def load_from_dir(cls, path):
        from bentoml.archive import load_bentoml_config

        if cls._bento_archive_path is not None and cls._bento_archive_path != path:
            logger.warning(
                "Loaded BentoArchive from '%s' can't be loaded again from a different"
                "path %s",
                cls._bento_archive_path,
                path,
            )

        artifacts_path = path
        # For pip installed BentoService, artifacts directory is located at
        # 'package_path/artifacts/', but for loading from BentoArchive, it is
        # in 'path/{service_name}/artifacts/'
        if not os.path.isdir(os.path.join(path, "artifacts")):
            artifacts_path = os.path.join(path, cls.name())

        bentoml_config = load_bentoml_config(path)
        if bentoml_config["metadata"]["service_name"] != cls.name():
            raise BentoMLException(
                "BentoService name does not match with BentoArchive in path: {}".format(
                    path
                )
            )

        if bentoml_config["kind"] != "BentoService":
            raise BentoMLException(
                "BentoArchive type '{}' can not be loaded as a BentoService".format(
                    bentoml_config["kind"]
                )
            )

        artifacts = ArtifactCollection.load(artifacts_path, cls._artifacts_spec)
        svc = cls(artifacts)
        return svc

    def _get_bento_service_metadata_pb(self):
        bento_service_metadata = BentoServiceMetadata()

        bento_service_metadata.name = self.name
        bento_service_metadata.version = self.version
        bento_service_metadata.created_at.GetCurrentTime()

        if self.env._setup_sh is not None:
            bento_service_metadata.env.setup_sh = self.env._setup_sh

        bento_service_metadata.env.conda_env = self.env._conda_env.to_yaml_str()
        bento_service_metadata.env.pip_dependencies = "\n".join(
            self.env._pip_dependencies
        )
        bento_service_metadata.env.python_version = self.env._python_version

        for api in self.get_service_apis():
            api_metadata = bento_service_metadata.apis.add()
            api_metadata.name = api.name
            api_metadata.handler_type = api.handler.__class__.__name__
            api_metadata.docs = api.doc

        for artifact_spec in self._artifacts_spec:
            artifact_metadata = bento_service_metadata.artifacts.add()
            artifact_metadata.name = artifact_spec.name
            artifact_metadata.artifact_type = artifact_spec.__class__.__name__

        return bento_service_metadata
