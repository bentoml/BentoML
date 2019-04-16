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
import re
import sys
import inspect
import importlib
from shutil import copyfile
from modulefinder import ModuleFinder

from six import string_types, iteritems

from bentoml.utils import Path
from bentoml.utils.exceptions import BentoMLException


def _get_module_src_file(module):
    """
    Return module.__file__, change extension to '.py' if __file__ is ending with '.pyc'
    """
    return module.__file__[:-1] if module.__file__.endswith('.pyc') else module.__file__


def _is_valid_py_identifier(s):
    """
    Return true if string is in a valid python identifier format:

    https://docs.python.org/2/reference/lexical_analysis.html#identifiers
    """
    return re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', s) is not None


def _get_module_relative_file_path(module_name, module_file):

    if not os.path.isabs(module_file):
        # For modules within current top level package, module_file here should
        # already be a relative path to the src file
        relative_path = module_file

    elif os.path.split(module_file)[1] == '__init__.py':
        # for module a.b.c in 'some_path/a/b/c/__init__.py', copy file to
        # 'destination/a/b/c/__init__.py'
        relative_path = os.path.join(module_name.replace('.', os.sep), '__init__.py')

    else:
        # for module a.b.c in 'some_path/a/b/c.py', copy file to 'destination/a/b/c.py'
        relative_path = os.path.join(module_name.replace('.', os.sep) + '.py')

    return relative_path


def copy_used_py_modules(target_module, destination):
    """
    bundle given module, and all its dependencies within top level package,
    and copy all source files to destination path, essentially creating
    a source distribution of target_module
    """

    # When target_module is a string, try import it
    if isinstance(target_module, string_types):
        try:
            target_module = importlib.import_module(target_module)
        except ImportError:
            pass
    target_module = inspect.getmodule(target_module)

    # When target module is defined in interactive session, we can not easily
    # get the class definition into a python module file and distribute it
    if target_module.__name__ == '__main__' and not hasattr(target_module, '__file__'):
        raise BentoMLException(
            "Custom BentoModel class can not be defined in Python interactive REPL, try "
            "writing the class definition to a file and import it.")

    try:
        target_module_name = target_module.__spec__.name
    except AttributeError:
        target_module_name = target_module.__name__

    target_module_file = _get_module_src_file(target_module)

    if target_module_name == '__main__':
        # Assuming no relative import in this case
        target_module_file_name = os.path.split(target_module_file)[1]
        target_module_name = target_module_file_name[:-3]  # remove '.py'

    # Find all modules must be imported for target module to run
    finder = ModuleFinder()
    # NOTE: This method could take a few seconds to run
    try:
        finder.run_script(target_module_file)
    except SyntaxError:
        # For package with conditional import that may only work with py2
        # or py3, ModuleFinder#run_script will try to compile the source
        # with current python version. And that may result in SyntaxError.
        pass

    # extra site-packages or dist-packages directory
    site_or_dist_package_path = [f for f in sys.path if f.endswith('-packages')]
    # prefix used to find installed Python library
    site_or_dist_package_path += [sys.prefix]
    # prefix used to find machine-specific Python library
    try:
        site_or_dist_package_path += [sys.base_prefix]
    except AttributeError:
        # ignore when in PY2 there is no sys.base_prefix
        pass

    # Look for dependencies that are not distributed python package, but users'
    # local python code, all other dependencies must be defined with @env
    # decorator when creating a new BentoService class
    user_packages_and_modules = {}
    for name, module in iteritems(finder.modules):
        if name == 'bentoml' or name.startswith('bentoml.'):
            # Remove BentoML library from dependent modules list
            break

        if hasattr(module, '__file__') and module.__file__ is not None:
            module_src_file = _get_module_src_file(module)

            is_module_in_site_or_dist_package = False
            for path in site_or_dist_package_path:
                if module_src_file.startswith(path):
                    is_module_in_site_or_dist_package = True
                    break

            if not is_module_in_site_or_dist_package:
                user_packages_and_modules[name] = module

    # Remove "__main__" module, if target module is loaded as __main__, it should
    # be in module_files as (module_name, module_file) in current context
    if '__main__' in user_packages_and_modules:
        del user_packages_and_modules['__main__']

    # Lastly, add target module itself
    user_packages_and_modules[target_module_name] = target_module

    for module_name, module in iteritems(user_packages_and_modules):
        module_file = _get_module_src_file(module)
        relative_path = _get_module_relative_file_path(module_name, module_file)
        target_file = os.path.join(destination, relative_path)

        # Create target directory if not exist
        Path(os.path.dirname(target_file)).mkdir(parents=True, exist_ok=True)

        # Copy module file to BentoArchive for distribution
        copyfile(module_file, target_file)

    for root, _, files in os.walk(destination):
        if '__init__.py' not in files:
            Path(os.path.join(root, '__init__.py')).touch()

    target_module_relative_path = _get_module_relative_file_path(target_module_name,
                                                                 target_module_file)

    return target_module_name, target_module_relative_path
