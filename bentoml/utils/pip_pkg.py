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


def collect_pip_dependencies(path):
    pass
