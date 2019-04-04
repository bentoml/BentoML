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
from abc import ABCMeta, abstractmethod
from six import add_metaclass

from bentoml.handlers import DataframeHandler


def _get_func_attr(func, attribute_name):
    if sys.version_info.major < 3 and inspect.ismethod(func):
        func = func.__func__
    return getattr(func, attribute_name)


def _set_func_attr(func, attribute_name, value):
    if sys.version_info.major < 3 and inspect.ismethod(func):
        func = func.__func__
    return setattr(func, attribute_name, value)


@add_metaclass(ABCMeta)
class BentoService(object):
    """
    BentoService is the base abstraction that exposes a list of APIs
    for BentoAPIServer and BentoCLI
    """

    @abstractmethod
    def load(self, path):
        """
        Load and initialize a BentoService
        """

    def _config_service_apis(self):
        self._apis = []
        for _, function in inspect.getmembers(
                self.__class__, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x)):
            if hasattr(function, '_is_api'):
                api_name = _get_func_attr(function, '_api_name')
                handler = _get_func_attr(function, '_handler')
                func = function.__get__(self)
                self._apis.append(BentoServiceAPI(api_name, handler, func))

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def version(self):
        pass

    @property
    def apis(self):
        return self._apis
        if not hasattr(self, '_service_apis'):
            self._config_service_apis()

    @staticmethod
    def api(handler=DataframeHandler, api_name=None):
        """
        Decorator for adding api to a BentoService

        >>> from bentoml import BentoService
        >>> from bentoml.handlers import JsonHandler, DataframeHandler
        >>>
        >>> class FraudDetectionAndIdentityService(BentoService):
        >>>
        >>>     @BentoService.api(JsonHandler)
        >>>     def fraud_detect(self, features):
        >>>         pass
        >>>
        >>>     @BentoService.api(DataframeHandler)
        >>>     def identity(self, features):
        >>>         pass
        """

        def api_decorator(func):
            _set_func_attr(func, '_is_api', True)
            _set_func_attr(func, '_handler', handler)
            # TODO: validate api_name
            if api_name is None:
                _set_func_attr(func, '_api_name', func.__name__)
            else:
                _set_func_attr(func, '_api_name', api_name)

            return func

        return api_decorator


class BentoServiceAPI(object):
    """
    BentoServiceAPI defines abstraction for an API call that can be executed
    with BentoAPIServer and BentoCLI
    """

    def __init__(self, name, handler, func):
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

    @property
    def name(self):
        return self._name

    @property
    def handler(self):
        return self._handler

    @property
    def func(self):
        return self._func


class SingleModelBentoService(BentoService):
    """
    A BentoService that host a single model's predict function as its only API
    """

    def __init__(self):
        # Adding BentoServiceAPI metadata to predict function
        try:
            handler = _get_func_attr(self.__class__.predict, "_handler")
        except AttributeError:
            handler = DataframeHandler
        BentoService.api(handler, 'predict')(self.__class__.predict)

        super(SingleModelBentoService, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def predict(self, data):
        pass


def handler_decorator(handler=DataframeHandler):
    """
    Decorator for setting handler Class for predict function in a BentoModel
    """

    def _handler_decorator(predict_func):
        if predict_func.__name__ != 'predict':
            raise ValueError("Bentoml's handler decoration can only "
                             "be applied to BentoModel#predict function")
        _set_func_attr(predict_func, '_handler', handler)
        return predict_func

    return _handler_decorator
