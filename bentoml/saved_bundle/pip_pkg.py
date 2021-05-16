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

import os
import sys
import logging
import pkgutil
import ast
import zipimport
import zipfile
from typing import Dict

EPP_NO_ERROR = 0
EPP_PKG_NOT_EXIST = 1
EPP_PKG_VERSION_MISMATCH = 2

ZIPIMPORT_DIR = 'zipimports'

__mm = None


logger = logging.getLogger(__name__)


def parse_requirement_string(rs):
    name, _, version = rs.partition("==")
    return name, version


def verify_pkg(pkg_req):
    global __mm  # pylint: disable=global-statement
    if __mm is None:
        __mm = ModuleManager()
    return __mm.verify_pkg(pkg_req)


def seek_pip_packages(target_py_file_path):
    global __mm  # pylint: disable=global-statement
    if __mm is None:
        __mm = ModuleManager()
    return __mm.seek_pip_packages(target_py_file_path)


def get_pkg_version(pkg_name):
    global __mm  # pylint: disable=global-statement
    if __mm is None:
        __mm = ModuleManager()
    return __mm.pip_pkg_map.get(pkg_name, None)


def get_zipmodules():
    global __mm  # pylint: disable=global-statement
    if __mm is None:
        __mm = ModuleManager()
    return __mm.zip_modules


def get_all_pip_installed_modules():
    global __mm  # pylint: disable=global-statement
    if __mm is None:
        __mm = ModuleManager()

    installed_modules = list(
        # local modules are the ones imported from current directory, either from a
        # module.py file or a module directory that contains a `__init__.py` file
        filter(lambda m: not m.is_local, __mm.searched_modules.values())
    )
    return list(map(lambda m: m.name, installed_modules))


class ModuleInfo(object):
    def __init__(self, name, path, is_local, is_pkg):
        super(ModuleInfo, self).__init__()
        self.name = name
        self.path = path
        self.is_local = is_local
        self.is_pkg = is_pkg


class ModuleManager(object):
    def __init__(self):
        super(ModuleManager, self).__init__()
        self.pip_pkg_map = {}
        self.pip_module_map = {}
        self.setuptools_module_set = set()
        self.nonlocal_package_path = set()

        import pkg_resources

        for dist in pkg_resources.working_set:  # pylint: disable=not-an-iterable
            module_path = dist.module_path or dist.location
            if not module_path:
                # Skip if no module path was found for pkg distribution
                continue

            if os.path.realpath(module_path) != os.getcwd():
                # add to nonlocal_package path only if it's not current directory
                self.nonlocal_package_path.add(module_path)

            self.pip_pkg_map[dist._key] = dist._version
            for mn in dist._get_metadata("top_level.txt"):
                if dist._key != "setuptools":
                    self.pip_module_map.setdefault(mn, []).append(
                        (dist._key, dist._version)
                    )
                else:
                    self.setuptools_module_set.add(mn)

        self.searched_modules = {}
        self.zip_modules: Dict[str, zipimport.zipimporter] = {}
        for m in pkgutil.iter_modules():
            if m.name not in self.searched_modules:
                if isinstance(m.module_finder, zipimport.zipimporter):
                    logger.info(f"Detected zipimporter {m.module_finder}")
                    path = m.module_finder.archive
                    self.zip_modules[path] = m.module_finder
                else:
                    path = m.module_finder.path
                is_local = self.is_local_path(path)
                self.searched_modules[m.name] = ModuleInfo(
                    m.name, path, is_local, m.ispkg
                )

    def verify_pkg(self, pkg_req):
        if pkg_req.name not in self.pip_pkg_map:
            # package does not exist in the current python session
            return EPP_PKG_NOT_EXIST

        if self.pip_pkg_map[pkg_req.name] not in pkg_req.specifier:
            # package version being used in the current python session does not meet
            # the specified package version requirement
            return EPP_PKG_VERSION_MISMATCH

        return EPP_NO_ERROR

    def seek_pip_packages(self, target_py_file_path):
        logger.debug('target py file path: %s', target_py_file_path)
        work = DepSeekWork(self, target_py_file_path)
        work.do()
        requirements = {}
        for _, pkg_info_list in work.dependencies.items():
            for pkg_name, pkg_version in pkg_info_list:
                requirements[pkg_name] = pkg_version

        return requirements, work.unknown_module_set

    def is_local_path(self, path):
        if path in self.nonlocal_package_path:
            return False

        dir_name = os.path.split(path)[1]

        if (
            "site-packages" in path
            or "anaconda" in path
            or path.endswith("packages")
            or dir_name == "bin"
            or dir_name.startswith("lib")
            or dir_name.startswith("python")
            or dir_name.startswith("plat")
        ):
            self.nonlocal_package_path.add(path)
            return False

        return True


class DepSeekWork(object):
    def __init__(self, module_manager, target_py_file_path):
        super(DepSeekWork, self).__init__()
        self.module_manager = module_manager
        self.target_py_file_path = target_py_file_path

        self.dependencies = {}
        self.unknown_module_set = set()
        self.parsed_module_set = set()

    def do(self):
        self.seek_in_file(self.target_py_file_path)

    def seek_in_file(self, file_path):
        try:
            with open(file_path) as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
        self.seek_in_source(content)

    def seek_in_source(self, content):
        # Extract all dependency modules by searching through the trees of the Python
        # abstract syntax grammar with Python's built-in ast module
        tree = ast.parse(content)
        import_set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    import_set.add(name.name.partition(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None and node.level == 0:
                    import_set.add(node.module.partition(".")[0])
        logger.debug("import set: %s", import_set)
        for module_name in import_set:
            # Avoid parsing BentoML when BentoML is imported from local source code repo
            if module_name == 'bentoml':
                continue
            if module_name in self.parsed_module_set:
                continue
            self.parsed_module_set.add(module_name)

            if module_name in self.module_manager.searched_modules:
                m = self.module_manager.searched_modules[module_name]
                if m.is_local:
                    # Recursively search dependencies in sub-modules
                    if m.path in self.module_manager.zip_modules:
                        self.seek_in_zip(m.path)
                    elif m.is_pkg:
                        self.seek_in_dir(os.path.join(m.path, m.name))
                    else:
                        self.seek_in_file(os.path.join(m.path, "{}.py".format(m.name)))
                else:
                    # check if the package has already been added to the list
                    if (
                        module_name in self.module_manager.pip_module_map
                        and module_name not in self.dependencies
                        and module_name not in self.module_manager.setuptools_module_set
                    ):
                        self.dependencies[
                            module_name
                        ] = self.module_manager.pip_module_map[module_name]
            else:
                if module_name in self.module_manager.pip_module_map:
                    if module_name not in self.dependencies:
                        # In some special cases, the pip-installed module can not
                        # be located in the searched_modules
                        self.dependencies[
                            module_name
                        ] = self.module_manager.pip_module_map[module_name]
                else:
                    if module_name not in sys.builtin_module_names:
                        self.unknown_module_set.add(module_name)

    def seek_in_dir(self, dir_path):
        for path, dir_list, file_list in os.walk(dir_path):
            for file_name in file_list:
                if not file_name.endswith(".py"):
                    continue
                self.seek_in_file(os.path.join(path, file_name))
            for dir_name in dir_list:
                if dir_name == '__pycache__':
                    continue
                self.seek_in_dir(os.path.join(path, dir_name))

    def seek_in_zip(self, zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            for module_path in zf.infolist():
                filename = module_path.filename
                if filename.endswith('.py'):
                    logger.debug("Seeking modules in zip %s", filename)
                    content = self.module_manager.zip_modules[zip_path].get_source(
                        filename.replace('.py', '')
                    )
                    self.seek_in_source(content)
