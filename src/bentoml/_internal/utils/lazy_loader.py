from __future__ import annotations

import sys
import types
import typing as t
import logging
import importlib

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
