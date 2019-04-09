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

import sys
import inspect

from bentoml.handlers import DataframeHandler
from bentoml.utils.exceptions import BentoMLException
from bentoml.service_env import BentoServiceEnv
from bentoml.artifacts import ArtifactCollection
from bentoml import archive


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

    def __init__(self, name, handler, func, options):
        """
        :param name: API name
        :param handler: A BentoHandler that transforms HTTP Request and/or
            CLI options into parameters for API func
        :param func: API func contains the actual API callback, this is
            typically the 'predict' method on a model
        """
        self._name = name
        self._handler = handler
        self._func = func
        self._options = options

    @property
    def name(self):
        return self._name

    @property
    def handler(self):
        return self._handler

    @property
    def func(self):
        return self._func

    @property
    def options(self):
        return self._options


class BentoService(object):
    """
    BentoService is the base abstraction that exposes a list of APIs
    for BentoAPIServer and BentoCLI to execute, and allow customizing
    the artifacts and environments required for the service.

    >>>  from bentoml import BentoService, env
    >>>  class MyMLService(BentoService):
    >>>
    >>>     artifacts = [PickleArtifact('clf')]
    >>>
    >>>     env = env(conda_dependencies: [ 'scikit-learn' ])
    >>>
    >>>     @BentoService.api(DataframeHandler)
    >>>     def predict(self, df):
    >>>         return self.artifacts.clf.predict(df)
    >>>
    >>>  bento_service = MyMLService.pack(clf=my_trained_clf_object)
    >>>  bentoml.save(bento_service, './export')
    """

    # User may override this if they don't want the generated model to
    # have the same name as their Python model class name
    __bento_service_name = None

    # This is overwritten when user install exported bento model as a
    # pip package, in that case, #load method will load from the installed
    # python package location
    __bento_module_path = None

    env = {}

    artifacts = []

    def __init__(self, artifacts, env=None):
        if isinstance(artifacts, ArtifactCollection):
            self._artifacts = artifacts
        else:
            self._artifacts = ArtifactCollection()
            for artifact in artifacts:
                self._artifacts[artifact.name] = artifact

        if env is None:
            if isinstance(self.__class__.env, dict):
                self._env = BentoServiceEnv.fromDict(self.__class__.env)
            else:
                self._env = self.__class__.env
        else:
            self._env = env

    @property
    def artifacts(self):
        return self._artifacts

    @property
    def env(self):
        return self._env

    @classmethod
    def name(cls):
        if cls.__bento_service_name is not None:
            # TODO: verify self.__class__.__bento_service_name format, can't have space in it
            #  and can be valid folder name
            return cls.__bento_service_name
        else:
            # Use python class name as service name
            return cls.__name__

    @property
    def name(self):
        return self.__class__.name()

    @property
    def version(self):
        try:
            return self._version
        except AttributeError:
            raise BentoMLException("Only BentoService loaded from archive has version attribute")

    def _config_service_apis(self):
        self._service_apis = []  # pylint:disable=attribute-defined-outside-init
        for _, function in inspect.getmembers(
                self.__class__, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x)):
            if hasattr(function, '_is_api'):
                api_name = _get_func_attr(function, '_api_name')
                handler = _get_func_attr(function, '_handler')
                func = function.__get__(self)
                options = _get_func_attr(function, '_options')
                self._service_apis.append(BentoServiceAPI(api_name, handler, func, options))

    def get_service_apis(self):
        if not hasattr(self, '_service_apis'):
            self._config_service_apis()
        return self._service_apis

    @staticmethod
    def api(handler=DataframeHandler, api_name=None, options=None):
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
        >>>     @api(DataframeHandler)
        >>>     def identity(self, df):
        >>>         # do something
        """

        def api_decorator(func):
            _set_func_attr(func, '_is_api', True)
            _set_func_attr(func, '_handler', handler)
            # TODO: validate api_name
            if api_name is None:
                _set_func_attr(func, '_api_name', func.__name__)
            else:
                _set_func_attr(func, '_api_name', api_name)

            _set_func_attr(func, '_options', options)

            return func

        return api_decorator

    @classmethod
    def pack(cls, *args, **kwargs):
        artifacts = ArtifactCollection()
        for artifact in cls.artifacts:
            artifacts[artifact.name] = artifact.create()
        artifacts.pack(*args, **kwargs)
        cls(artifacts)

    def save(self, *args, **kwargs):
        return archive.save(self, *args, **kwargs)

    @classmethod
    def load(cls, path):
        return archive.load(cls, path)