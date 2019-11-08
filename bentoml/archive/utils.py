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
from __future__ import division
from __future__ import print_function

import os
import shutil
import importlib
from setuptools import sandbox


def _find_bentoml_module_location():
    try:
        module_location, = importlib.util.find_spec(
            'bentoml'
        ).submodule_search_locations
    except AttributeError:
        # python 2.7 doesn't have importlib.util, will fall back to imp instead
        import imp

        _, module_location, _ = imp.find_module('bentoml')
    return module_location


def add_local_bentoml_package_to_repo(archive_path):
    module_location = _find_bentoml_module_location()
    bentoml_setup_py = os.path.abspath(os.path.join(module_location, '..', 'setup.py'))

    assert os.path.isfile(bentoml_setup_py), '"setup.py" for Bentoml module not found'

    # Create tmp directory inside bentoml module for storing the bundled
    # targz file. Since dist-dir can only be inside of the module directory
    bundle_dir_name = '__bento_dev_tmp'
    source_dir = os.path.abspath(os.path.join(module_location, '..', bundle_dir_name))
    if os.path.isdir(source_dir):
        shutil.rmtree(source_dir, ignore_errors=True)
    os.mkdir(source_dir)

    sandbox.run_setup(
        bentoml_setup_py, ['sdist', '--format', 'gztar', '--dist-dir', bundle_dir_name]
    )

    # copy the generated targz to archive directory and remove it from
    # bentoml module directory
    dest_dir = os.path.join(archive_path, 'bundled_pip_dependencies')
    shutil.copytree(source_dir, dest_dir)
    shutil.rmtree(source_dir)
