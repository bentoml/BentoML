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
import ast
import pkg_resources

__pip_pkg_map = {}
__pip_module_map = {}

EPP_NO_ERROR = 0
EPP_PKG_NOT_EXIST = 1
EPP_PKG_VERSION_MISMATCH = 2


def collect_pip_pkg_info():
    for dist in pkg_resources.working_set:
        __pip_pkg_map[dist._key] = dist._version
        for mn in dist._get_metadata("top_level.txt"):
            __pip_module_map.setdefault(mn, []).append((dist._key, dist._version))


def parse_requirement_string(rs):
    name, _, version = rs.partition("==")
    return name, version


def verify_pkg(pkg_name, pkg_version):
    if not __pip_pkg_map:
        collect_pip_pkg_info()
    if pkg_name not in __pip_pkg_map:
        # package does not exist in the current python session
        return EPP_PKG_NOT_EXIST
    if pkg_version and pkg_version != __pip_pkg_map[pkg_name]:
        # package version is different from the version being used
        # in the current python session
        return EPP_PKG_VERSION_MISMATCH
    return EPP_NO_ERROR


def seek_pip_dependencies(root_path):
    root_module = parse_dir_module(root_path, None)
    external_import_set = extract_external_import_set(root_module)
    return filter_requirements(external_import_set, root_path)


class Module(object):

    def __init__(self, name, parent):
        super(Module, self).__init__()
        self.name = name
        self.parent = parent
        self.sub_module_map = {}   # name -> module
        self.import_set = set()

    def add_sub_module(self, sub_mod):
        self.sub_module_map[sub_mod.name] = sub_mod

    def __contains__(self, item):
        return item in self.sub_module_map

    def sub_modules(self):
        return self.sub_module_map.values()


def parse_file_module(content, module_name, parent):
    """

    :param content: py file content
    :param module_name: py file name, as module name
    :param parent: parent module
    :return: py file module
    """
    mod = Module(module_name, parent)
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                mod.import_set.add(name.name.partition(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:
                mod.import_set.add(node.module.partition(".")[0])
    return mod


def parse_dir_module(path, parent):
    """

    :param path: package path
    :param parent: parent module
    :return: package module
    """
    module_name = os.path.basename(path)
    mod = Module(module_name, parent)
    for item in os.listdir(path):
        if item.startswith("."):
            continue

        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            sub_mod = parse_dir_module(item_path, mod)
            mod.add_sub_module(sub_mod)
        else:
            name, ext = os.path.splitext(item)
            if ext != ".py":
                continue

            with open(item_path) as f:
                sub_mod = parse_file_module(f.read(), name, mod)
                mod.add_sub_module(sub_mod)

    return mod


def extract_external_import_set(root_module):
    external_import_set = set()
    mod_list = [root_module]
    while mod_list:
        mod = mod_list.pop(0)
        for import_m in mod.import_set:
            if import_m not in root_module:
                external_import_set.add(import_m)
        mod_list.extend(mod.sub_modules())
    return external_import_set


def filter_requirements(external_import_set, project_path):
    # pip安装的包
    if not __pip_module_map:
        collect_pip_pkg_info()

    # 已安装的模块，包含自带的包和第三方安装的包（第三方安装的包包含pip安装的包）
    installed_modules = collect_installed_modules(project_path)
    # 内建的模块
    builtin_modules = sys.builtin_module_names
    requirements = []
    unknown_modules = []
    for module_name in external_import_set:
        if module_name in __pip_module_map:
            requirements.extend(__pip_module_map[module_name])
        else:
            if module_name not in installed_modules \
               and module_name not in builtin_modules:
                unknown_modules.append(module_name)
    return requirements, unknown_modules


def collect_installed_modules(project_path):
    import pkgutil
    installed_modules = {}
    for m in pkgutil.iter_modules():
        if getattr(m[0], "path", "") == project_path:
            continue
        installed_modules[m[1]] = m[2]
    return installed_modules
