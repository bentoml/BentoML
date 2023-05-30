from __future__ import annotations

import os
import sys
import types
import typing as t
import logging
import importlib
import itertools
import importlib.machinery

from ...exceptions import MissingDependencyException

logger = logging.getLogger(__name__)


class LazyLoader(types.ModuleType):
    """
    LazyLoader module borrowed from Tensorflow
     https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/util/lazy_loader.py
     with a addition of "module caching". This will throw an
     exception if module cannot be imported.

    Lazily import a module, mainly to avoid pulling in large dependencies.
     `contrib`, and `ffmpeg` are examples of modules that are large and not always
     needed, and this allows them to only be loaded when they are used.
    """

    def __init__(
        self,
        local_name: str,
        parent_module_globals: dict[str, t.Any],
        name: str,
        warning: str | None = None,
        exc_msg: str | None = None,
        exc: type[Exception] = MissingDependencyException,
    ):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._warning = warning
        self._exc_msg = exc_msg
        self._exc = exc
        self._module: types.ModuleType | None = None

        super().__init__(str(name))

    def _load(self) -> types.ModuleType:
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        try:
            module = importlib.import_module(self.__name__)
            self._parent_module_globals[self._local_name] = module
            # The additional add to sys.modules ensures library is actually loaded.
            sys.modules[self._local_name] = module
        except ModuleNotFoundError as err:
            raise self._exc(f"{self._exc_msg} (reason: {err})") from None

        # Emit a warning if one was specified
        if self._warning:
            logger.warning(self._warning)
            # Make sure to only warn once.
            self._warning = None

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: t.Any) -> t.Any:
        if self._module is None:
            self._module = self._load()
        return getattr(self._module, item)

    def __dir__(self) -> t.List[str]:
        if self._module is None:
            self._module = self._load()
        return dir(self._module)


class LazyModule(types.ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    This is a port from transformers.utils.import_utils._LazyModule for backwards compatibility with transformers < 4.18

    This is an extension a more powerful LazyLoader.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: dict[str, list[str]],
        module_spec: importlib.machinery.ModuleSpec | None = None,
        extra_objects: dict[str, t.Any] | None = None,
    ):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module: dict[str, str] = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(
            itertools.chain(*import_structure.values())
        )
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = t.cast("list[str]", super().__dir__())
        # The elements of self.__all__ that are submodules may or
        # may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the
        # elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> t.Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))
