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

from __future__ import absolute_import

import os
import sys
import pkg_resources
import pkgutil
import ast

EPP_NO_ERROR = 0
EPP_PKG_NOT_EXIST = 1
EPP_PKG_VERSION_MISMATCH = 2

__mm = None


def parse_requirement_string(rs):
    name, _, version = rs.partition("==")
    return name, version


def verify_pkg(pkg_name, pkg_version):
    global __mm
    if __mm is None:
        __mm = ModuleManager()
    return __mm.verify_pkg(pkg_name, pkg_version)


def seek_pip_dependencies(target_py_file_path):
    global __mm
    if __mm is None:
        __mm = ModuleManager()
    return __mm.seek_pip_dependencies(target_py_file_path)


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
        for dist in pkg_resources.working_set:
            self.nonlocal_package_path.add(dist.module_path)
            self.pip_pkg_map[dist._key] = dist._version
            for mn in dist._get_metadata("top_level.txt"):
                if dist._key != "setuptools":
                    self.pip_module_map.setdefault(mn, []).append(
                        (dist._key, dist._version)
                    )
                else:
                    self.setuptools_module_set.add(mn)

        self.searched_modules = {}
        for m in pkgutil.iter_modules():
            if m.name not in self.searched_modules:
                path = m.module_finder.path
                is_local = self.is_local_path(path)
                self.searched_modules[m.name] = ModuleInfo(
                    m.name, path, is_local, m.ispkg
                )

    def verify_pkg(self, pkg_name, pkg_version):
        if pkg_name not in self.pip_pkg_map:
            # package does not exist in the current python session
            return EPP_PKG_NOT_EXIST
        if pkg_version and pkg_version != self.pip_pkg_map[pkg_name]:
            # package version is different from the version being used
            # in the current python session
            return EPP_PKG_VERSION_MISMATCH
        return EPP_NO_ERROR

    def seek_pip_dependencies(self, target_py_file_path):
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
        # ast解析py file，获取其依赖的modules
        with open(file_path) as f:
            content = f.read()
            tree = ast.parse(content)
            import_set = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        import_set.add(name.name.partition(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module is not None and node.level == 0:
                        import_set.add(node.module.partition(".")[0])
            for module_name in import_set:
                if module_name in self.parsed_module_set:
                    continue
                self.parsed_module_set.add(module_name)

                if module_name in self.module_manager.searched_modules:
                    m = self.module_manager.searched_modules[module_name]
                    if m.is_local:
                        # 递归解析
                        if m.is_pkg:
                            self.seek_in_dir(os.path.join(m.path, m.name))
                        else:
                            self.seek_in_file(
                                os.path.join(m.path, "{}.py".format(m.name))
                            )
                    else:
                        # 判断是否在pip安装包中
                        if (
                            module_name in self.module_manager.pip_module_map
                            and module_name not in self.dependencies
                            and module_name
                            not in self.module_manager.setuptools_module_set
                        ):
                            self.dependencies[
                                module_name
                            ] = self.module_manager.pip_module_map[module_name]
                else:
                    if module_name in self.module_manager.pip_module_map:
                        if module_name not in self.dependencies:
                            # 某些特殊情况下，pip安装的module不存在searched_modules中
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
