import os
import re
import sys
import inspect
import logging
import importlib
import modulefinder
from typing import List
from typing import Tuple
from unittest.mock import patch

from ..types import PathType
from .pip_pkg import get_all_pip_installed_modules
from ...exceptions import BentoMLException

logger = logging.getLogger(__name__)


def _get_module_src_file(module):
    """
    Return module.__file__, change extension to '.py' if __file__ is ending with '.pyc'
    """
    return module.__file__[:-1] if module.__file__.endswith(".pyc") else module.__file__


def _is_valid_py_identifier(s):
    """
    Return true if string is in a valid python identifier format:

    https://docs.python.org/2/reference/lexical_analysis.html#identifiers
    """
    return re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", s) is not None


def _get_module_relative_file_path(module_name, module_file):

    if not os.path.isabs(module_file):
        # For modules within current top level package, module_file here should
        # already be a relative path to the src file
        relative_path = module_file

    elif os.path.split(module_file)[1] == "__init__.py":
        # for module a.b.c in 'some_path/a/b/c/__init__.py', copy file to
        # 'destination/a/b/c/__init__.py'
        relative_path = os.path.join(module_name.replace(".", os.sep), "__init__.py")

    else:
        # for module a.b.c in 'some_path/a/b/c.py', copy file to 'destination/a/b/c.py'
        relative_path = os.path.join(module_name.replace(".", os.sep) + ".py")

    return relative_path


def _get_module(target_module):
    # When target_module is a string, try import it
    if isinstance(target_module, str):
        try:
            target_module = importlib.import_module(target_module)
        except ImportError:
            pass
    return inspect.getmodule(target_module)


def _import_module_from_file(path):
    module_name = path.replace(os.sep, ".")[:-3]
    spec = importlib.util.spec_from_file_location(module_name, path)
    m = importlib.util.module_from_spec(spec)
    return m


# TODO: change this to find_local_py_modules_used(svc: Service)
def find_local_py_modules_used(target_module_file: PathType) -> List[Tuple[str, str]]:
    """Find all local python module dependencies of target_module, and list all the
    local module python files to the destination directory while maintaining the module
    structure unchanged to ensure all imports in target_module still works when loading
    from the destination directory again

    Args:
       path (`Union[str, bytes, os.PathLike]`):
            Path to a python source file

       Returns:
            list of (source file path, target file path) pairs
    """

    target_module = _import_module_from_file(target_module_file)

    try:
        target_module_name = target_module.__spec__.name
    except AttributeError:
        target_module_name = target_module.__name__

    # Find all non pip installed modules must be packaged for target module to run
    exclude_modules = ["bentoml"] + get_all_pip_installed_modules()
    finder = modulefinder.ModuleFinder(excludes=exclude_modules)

    try:
        logger.debug(
            "Searching for local dependant modules of %s:%s",
            target_module_name,
            target_module_file,
        )
        if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
            _find_module = modulefinder._find_module
            _PKG_DIRECTORY = modulefinder._PKG_DIRECTORY

            def _patch_find_module(name, path=None):
                """ref issue: https://bugs.python.org/issue40350"""

                importlib.machinery.PathFinder.invalidate_caches()

                spec = importlib.machinery.PathFinder.find_spec(name, path)

                if spec is not None and spec.loader is None:
                    return None, None, ("", "", _PKG_DIRECTORY)

                return _find_module(name, path)

            with patch.object(modulefinder, "_find_module", _patch_find_module):
                finder.run_script(target_module_file)
        else:
            finder.run_script(target_module_file)
    except SyntaxError:
        # For package with conditional import that may only work with py2
        # or py3, ModuleFinder#run_script will try to compile the source
        # with current python version. And that may result in SyntaxError.
        pass

    if finder.badmodules:
        logger.debug(
            "Find bad module imports that can not be parsed properly: %s",
            finder.badmodules.keys(),
        )

    # Look for dependencies that are not distributed python package, but users'
    # local python code, all other dependencies must be defined with @env
    # decorator when creating a new BentoService class
    user_packages_and_modules = {}
    for name, module in finder.modules.items():
        if hasattr(module, "__file__") and module.__file__ is not None:
            user_packages_and_modules[name] = module

    # Lastly, add target module itself
    user_packages_and_modules[target_module_name] = target_module

    file_list = []
    for module_name, module in user_packages_and_modules.items():
        module_file = _get_module_src_file(module)
        relative_path = _get_module_relative_file_path(module_name, module_file)
        file_list.append((module_file, relative_path))

    return file_list


def copy_local_py_modules(target_module, destination):
    """Find all local python module dependencies of target_module, and copy all the
    local module python files to the destination directory while maintaining the module
    structure unchanged to ensure all imports in target_module still works when loading
    from the destination directory again
    """
    target_module = _get_module(target_module)

    # When target module is defined in interactive session, we can not easily
    # get the class definition into a python module file and distribute it
    if target_module.__name__ == "__main__" and not hasattr(target_module, "__file__"):
        raise BentoMLException(
            "Custom BentoModel class can not be defined in Python interactive REPL, try"
            " writing the class definition to a file and import it."
        )

    try:
        target_module_name = target_module.__spec__.name
    except AttributeError:
        target_module_name = target_module.__name__

    target_module_file = _get_module_src_file(target_module)
    logger.debug(
        "copy_local_py_modules target_module_name: %s, target_module_file: %s",
        target_module_name,
        target_module_file,
    )

    if target_module_name == "__main__":
        # Assuming no relative import in this case
        target_module_file_name = os.path.split(target_module_file)[1]
        target_module_name = target_module_file_name[:-3]  # remove '.py'
        logger.debug(
            "Updating for __main__ module, target_module_name: %s, "
            "target_module_file: %s",
            target_module_name,
            target_module_file,
        )

    # Find all non pip installed modules must be packaged for target module to run
    # exclude_modules = ['bentoml'] + get_all_pip_installed_modules()
    # finder = modulefinder.ModuleFinder(excludes=exclude_modules)
    #
    # try:
    #     logger.debug(
    #         "Searching for local dependant modules of %s:%s",
    #         target_module_name,
    #         target_module_file,
    #     )
    #     if sys.version_info[0] == 3 and sys.version_info[1] >= 8:
    #         _find_module = modulefinder._find_module
    #         _PKG_DIRECTORY = modulefinder._PKG_DIRECTORY
    #
    #         def _patch_find_module(name, path=None):
    #             """ref issue: https://bugs.python.org/issue40350"""
    #
    #             importlib.machinery.PathFinder.invalidate_caches()
    #
    #             spec = importlib.machinery.PathFinder.find_spec(name, path)
    #
    #             if spec is not None and spec.loader is None:
    #                 return None, None, ("", "", _PKG_DIRECTORY)
    #
    #             return _find_module(name, path)
    #
    #         with patch.object(modulefinder, '_find_module', _patch_find_module):
    #             finder.run_script(target_module_file)
    #     else:
    #         finder.run_script(target_module_file)
    # except SyntaxError:
    #     # For package with conditional import that may only work with py2
    #     # or py3, ModuleFinder#run_script will try to compile the source
    #     # with current python version. And that may result in SyntaxError.
    #     pass
    #
    # if finder.badmodules:
    #     logger.debug(
    #         "Find bad module imports that can not be parsed properly: %s",
    #         finder.badmodules.keys(),
    #     )
    #
    # # Look for dependencies that are not distributed python package, but users'
    # # local python code, all other dependencies must be defined with @env
    # # decorator when creating a new BentoService class
    # user_packages_and_modules = {}
    # for name, module in finder.modules.items():
    #     if hasattr(module, "__file__") and module.__file__ is not None:
    #         user_packages_and_modules[name] = module

    # # Remove "__main__" module, if target module is loaded as __main__, it should
    # # be in module_files as (module_name, module_file) in current context
    # if "__main__" in user_packages_and_modules:
    #     del user_packages_and_modules["__main__"]
    #
    # # Lastly, add target module itself
    # user_packages_and_modules[target_module_name] = target_module
    # logger.debug(
    #     "Copying user local python dependencies: %s", user_packages_and_modules
    # )
    #
    # for module_name, module in user_packages_and_modules.items():
    #     module_file = _get_module_src_file(module)
    #     relative_path = _get_module_relative_file_path(module_name, module_file)
    #     target_file = os.path.join(destination, relative_path)
    #
    #     # Create target directory if not exist
    #     Path(os.path.dirname(target_file)).mkdir(parents=True, exist_ok=True)
    #
    #     # Copy module file to BentoArchive for distribution
    #     logger.debug("Copying local python module '%s'", module_file)
    #     copyfile(module_file, target_file)
    #
    # for root, _, files in os.walk(destination):
    #     if "__init__.py" not in files:
    #         logger.debug("Creating empty __init__.py under folder:'%s'", root)
    #         Path(os.path.join(root, "__init__.py")).touch()
    #
    # target_module_relative_path = _get_module_relative_file_path(
    #     target_module_name, target_module_file
    # )
    # logger.debug("Done copying local python dependant modules")
    #
    # return target_module_name, target_module_relative_path
