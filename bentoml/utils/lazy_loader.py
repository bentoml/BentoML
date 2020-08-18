import importlib
import types
import logging

logger = logging.getLogger(__name__)


class LazyLoader(types.ModuleType):
    """LazyLoader module borrowed from Tensorflow
  https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/util/lazy_loader.py

  Lazily import a module, mainly to avoid pulling in large dependencies.
  `contrib`, and `ffmpeg` are examples of modules that are large and not always
  needed, and this allows them to only be loaded when they are used.
  """

    # The lint error here is incorrect.
    def __init__(self, local_name, parent_module_globals, name, warning=None):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._warning = warning

        super(LazyLoader, self).__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

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

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
