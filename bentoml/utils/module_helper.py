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
import inspect
import shutil
import importlib
from modulefinder import ModuleFinder

from six import string_types, iteritems

from bentoml.utils import Path
from bentoml.utils.exceptions import BentoMLException


def _get_module_src_file(module):
    return module.__file__[:-1] if module.__file__.endswith('.pyc') else module.__file__


def copy_module_and_local_dependencies(target_module, destination, toplevel_package_path=None,
                                       copy_entire_package=False):
    """
    bundle given module, and all its dependencies within top level package,
    and copy all source files to destination path, essentially creating
    a source distribution of target_module
    """
    if isinstance(target_module, string_types):
        target_module = importlib.import_module(target_module)
    else:
        target_module = inspect.getmodule(target_module)

    if target_module.__name__ == '__main__' and not target_module.__file__:
        raise BentoMLException(
            "Custom BentoModel class can not be defined in Python interactive REPL, try "
            "writing the class definition to a file and import, e.g. my_bentoml_model.py")

    try:
        target_module_name = target_module.__spec__.name
    except AttributeError:
        target_module_name = target_module.__name__

    target_module_file = _get_module_src_file(target_module)

    if target_module_name == '__main__':
        target_module_name = target_module_file[:-3].replace(os.sep, '.')

    if copy_entire_package:
        if toplevel_package_path is None:
            raise BentoMLException("Must set toplevel_package_path when using"
                                   "copy_entire_package=True")
        shutil.copytree(toplevel_package_path, destination)
        return target_module_name, target_module_file

    # Try use current directory or top level package directory path as project base path
    # This will be used to determine if a module is loacl dependency that should be copied
    # into bentoml model archive
    if toplevel_package_path is None:
        toplevel_package_name = target_module_name.split('.')[0]
        toplevel_package = importlib.import_module(
            toplevel_package_name)  # Should already loaded in sys.modules

        if hasattr(toplevel_package, '__path__'):
            toplevel_package_path_list = map(lambda path: os.path.join(path, '..'),
                                             toplevel_package.__path__)
        else:
            toplevel_package_path_list = [
                os.path.join(os.path.dirname(toplevel_package.__file__), '..')
            ]
    else:
        toplevel_package_path_list = [toplevel_package_path]
    toplevel_package_path_list = map(os.path.abspath, toplevel_package_path_list)

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

    # Remove dependencies not in current project source code
    # all third party dependencies must be defined in BentoEnv when creating model
    local_modules = {}
    for path in toplevel_package_path_list:
        for name, module in iteritems(finder.modules):
            if module.__file__ is not None:
                if name not in local_modules and _get_module_src_file(module).startswith(path):
                    local_modules[name] = module

    # Remove "__main__" module, if target module is loaded as __main__, it should
    # be in module_files as (module_name, module_file) in current context
    if '__main__' in local_modules:
        del local_modules['__main__']

    # Lastly, add target module itself
    local_modules[target_module_name] = target_module

    for module_name, module in iteritems(local_modules):
        module_file = _get_module_src_file(module)

        with open(module_file, "rb") as f:
            src_code = f.read()

        if not os.path.isabs(module_file):
            # For modules within current top level package, module_file here should be a
            # relative path to the src file
            target_file = os.path.join(destination, module_file)
        elif os.path.split(module_file)[1] == '__init__.py':
            # for module a.b.c in 'some_path/a/b/c/__init__.py', copy file to
            # 'destination/a/b/c/__init__.py'
            target_file = os.path.join(destination, module_name.replace('.', os.sep), '__init__.py')
        else:
            # for module a.b.c in 'some_path/a/b/c.py', copy file to 'destination/a/b/c.py'
            target_file = os.path.join(destination, module_name.replace('.', os.sep) + '.py')

        target_path = os.path.dirname(target_file)
        Path(target_path).mkdir(parents=True, exist_ok=True)

        with open(target_file, 'wb') as f:
            f.write(src_code)

    for root, _, files in os.walk(destination):
        if '__init__.py' not in files:
            Path(os.path.join(root, '__init__.py')).touch()

    return target_module_name, target_module_file
