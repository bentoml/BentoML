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


class hybridmethod(object):
    """A decorator which allows definition of a Python object method with both
    instance-level and class-level behavior.

    """

    def __init__(self, instance_function):
        """Create a new :class:`.hybridmethod`.

       Usage is typically via decorator::

           from bentoml.util.hybridmethod import hybridmethod

           class SomeClass(object):
               @hybridmethod
               def func(self, x, y):
                   return self._value + x + y

               @func.classmethod
               def value(cls, x, y):
                   return cls._default_value + x + y
       """
        self._instance_function = instance_function
        self._class_function = None

    def __get__(self, instance, owner):
        if instance is None:
            assert self._class_function is not None, "classmethod not defined"
            return self._class_function.__get__(owner, type(owner))
        else:
            return self._instance_function.__get__(instance, owner)

    def classmethod(self, class_function):
        self._class_function = class_function
        if not self._class_function.__doc__:
            self._class_function.__doc__ = self._instance_function.__doc__
        return self

    def __call__(self, *args, **kwargs):
        """Needed for this to work in python2.7
        """
        return self.__get__(self, type(self))(*args, **kwargs)
