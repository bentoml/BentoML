# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import inspect
import tempfile

from six import add_metaclass
from abc import abstractmethod, ABCMeta

from bentoml.utils.exceptions import BentoMLException
from bentoml.service_env import BentoServiceEnv
from bentoml.artifact import ArtifactCollection
from bentoml.utils.s3 import is_s3_url, download_from_s3


def _get_func_attr(func, attribute_name):
    if sys.version_info.major < 3 and inspect.ismethod(func):
        func = func.__func__
    return getattr(func, attribute_name)


def _set_func_attr(func, attribute_name, value):
    if sys.version_info.major < 3 and inspect.ismethod(func):
        func = func.__func__
    return setattr(func, attribute_name, value)


# TODO(chaoyu): add property info, default to api func's doc string
class BentoServiceAPI(object):
    """
    BentoServiceAPI defines abstraction for an API call that can be executed
    with BentoAPIServer and BentoCLI
    """

    def __init__(self, service, name, handler, func):
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
        self._handler = handler
        self._func = func

    @property
    def service(self):
        return self._service

    @property
    def name(self):
        return self._name

    @property
    def handler(self):
        return self._handler

    @property
    def func(self):
        return self._func

    def handle_request(self, request):
        return self.handler.handle_request(request, self.func)

    def handle_cli(self, args):
        return self.handler.handle_cli(args, self.func)

    def handle_aws_lambda_event(self, event):
        return self.handler.handle_aws_lambda_event(event, self.func)


@add_metaclass(ABCMeta)
class BentoServiceBase(object):
    """
    BentoServiceBase is the base abstraction that exposes a list of APIs
    for BentoAPIServer and BentoCLI to execute
    """

    @property
    @abstractmethod
    def name(self):
        """
        :return bento service name
        """

    @property
    @abstractmethod
    def version(self):
        """
        :return bento service version str
        """

    def _config_service_apis(self):
        self._service_apis = []  # pylint:disable=attribute-defined-outside-init
        for _, function in inspect.getmembers(
                self.__class__, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x)):
            if hasattr(function, '_is_api'):
                api_name = _get_func_attr(function, '_api_name')
                handler = _get_func_attr(function, '_handler')

                # Bind api method call with self(BentoService instance)
                func = function.__get__(self)

                self._service_apis.append(BentoServiceAPI(self, api_name, handler, func))

    def get_service_apis(self):
        return self._service_apis


def api_decorator(handler_cls, *args, **kwargs):
    """
    Decorator for adding api to a BentoService

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

    def decorator(func):
        api_name = kwargs.pop('api_name', func.__name__)
        handler = handler_cls(*args, **kwargs)  # create handler instance and attach to api method

        _set_func_attr(func, '_is_api', True)
        _set_func_attr(func, '_handler', handler)
        # TODO: validate api_name
        _set_func_attr(func, '_api_name', api_name)

        return func

    return decorator


def artifacts_decorator(artifact_specs):

    def decorator(bento_service_cls):
        bento_service_cls._artifacts_spec = artifact_specs
        return bento_service_cls

    return decorator


def env_decorator(**kwargs):

    def decorator(bento_service_cls):
        bento_service_cls._env = BentoServiceEnv.fromDict(kwargs)
        return bento_service_cls

    return decorator


def ver_decorator(major, minor):
    """Decorator for specifying the version of a custom BentoService, BentoML
    uses semantic versioning for BentoService distribution:

    MAJOR is incremented when you make breaking API changes
    MINOR is incremented when you add new functionality without breaking the
        existing API or functionality
    PATCH is incremented when you make backwards-compatible bug fixes

    'Patch' is provided(or auto generated) when calling BentoService#save,
    while 'Major' and 'Minor' can be defined with '@ver' decorator

    >>>  @ver(major=1, minor=4)
    >>>  @artifacts([PickleArtifact('model')])
    >>>  class MyMLService(BentoService):
    >>>     pass
    >>>
    >>>  svc = MyMLService.pack(model="my ML model object")
    >>>  svc.save('/path_to_archive', version="2019-08.iteration20")
    >>>  # The final produced BentoArchive verion will be "1.4.2019-08.iteration20"
    """

    def decorator(bento_service_cls):
        bento_service_cls._version_major = major
        bento_service_cls._version_minor = minor
        return bento_service_cls

    return decorator


class BentoService(BentoServiceBase):
    """
    BentoService packs a list of artifacts and exposes service APIs
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
    >>>  bentoml.save(bento_service, './export')
    """

    # User may override this if they don't want the generated model to
    # have the same name as their Python model class name
    _bento_service_name = None

    # For BentoService loaded from achive directory, this will be set to archive path
    # when user install exported bento model as a PyPI package, this will be set to
    # the python installed site-package location
    _bento_archive_path = None

    # list of artifact spec describing required artifacts for this BentoService
    _artifacts_spec = []

    # Describe the desired environment for this BentoService using
    # `bentoml.service_env.BentoServiceEnv`
    _env = {}

    # This can only be set by BentoML library when loading from archive
    _bento_service_version = None

    # See `ver_decorator` function above for more information
    _version_major = None
    _version_minor = None

    def __init__(self, artifacts=None, env=None):
        if artifacts is None:
            if self._bento_archive_path:
                artifacts = ArtifactCollection.load(self._bento_archive_path,
                                                    self.__class__._artifacts_spec)
            else:
                raise BentoMLException("Must provide artifacts or set cls._bento_archive_path"
                                       "before instantiating a BentoService class")

        # TODO: validate artifacts arg matches self.__class__._artifacts_spec definition
        if isinstance(artifacts, ArtifactCollection):
            self._artifacts = artifacts
        else:
            self._artifacts = ArtifactCollection()
            for artifact in artifacts:
                self._artifacts[artifact.name] = artifact

        self._init_env(env)
        self._config_service_apis()
        self.name = self.__class__.name()

    def _init_env(self, env=None):
        if env is None:
            # By default use BentoServiceEnv defined on class via @env decorator
            env = self.__class__._env

        if isinstance(env, dict):
            self._env = BentoServiceEnv.fromDict(env)
        else:
            self._env = env

    @property
    def artifacts(self):
        return self._artifacts

    @property
    def env(self):
        return self._env

    @classmethod
    def name(cls):  # pylint:disable=method-hidden
        if cls._bento_service_name is not None:
            # TODO: verify self.__class__._bento_service_name format, can't have space in it
            #  and can be valid folder name
            return cls._bento_service_name
        else:
            # Use python class name as service name
            return cls.__name__

    @property
    def version(self):
        try:
            return self.__class__._bento_service_version
        except AttributeError:
            raise BentoMLException("Only BentoService loaded from archive has version attribute")

    def save(self, *args, **kwargs):
        from bentoml import archive
        return archive.save(self, *args, **kwargs)

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
    def from_archive(cls, path):
        from bentoml.archive import load_bentoml_config

        # TODO: add model.env.verify() to check dependencies and python version etc
        if cls._bento_archive_path is not None and cls._bento_archive_path != path:
            raise BentoMLException(
                "Loaded BentoArchive(from {}) can't be loaded again from a different"
                "archive path {}".format(cls._bento_archive_path, path))

        if is_s3_url(path):
            temporary_path = tempfile.mkdtemp()
            download_from_s3(path, temporary_path)
            # Use loacl temp path for the following loading operations
            path = temporary_path

        artifacts_path = path
        # For pip installed BentoService, artifacts directory is located at
        # 'package_path/artifacts/', but for loading from BentoArchive, it is
        # in 'path/{service_name}/artifacts/'
        if not os.path.isdir(os.path.join(path, 'artifacts')):
            artifacts_path = os.path.join(path, cls.name())

        bentoml_config = load_bentoml_config(path)
        # TODO: check archive type and allow loading archive only
        if bentoml_config['service_name'] != cls.name():
            raise BentoMLException(
                'BentoService name does not match with BentoML Archive in path: {}'.format(path))

        artifacts = ArtifactCollection.load(artifacts_path, cls._artifacts_spec)
        svc = cls(artifacts)
        return svc
