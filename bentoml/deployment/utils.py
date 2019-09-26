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

import json
import os
import logging
import importlib
import shutil

from setuptools import sandbox

from bentoml.utils import Path

logger = logging.getLogger(__name__)


def process_docker_api_line(payload):
    """ Process the output from API stream, throw an Exception if there is an error """
    # Sometimes Docker sends to "{}\n" blocks together...
    for segment in payload.decode("utf-8").split("\n"):
        line = segment.strip()
        if line:
            try:
                line_payload = json.loads(line)
            except ValueError as e:
                print("Could not decipher payload from Docker API: " + e)
            if line_payload:
                if "errorDetail" in line_payload:
                    error = line_payload["errorDetail"]
                    logger.error(error['message'])
                    raise RuntimeError(
                        "Error on build - code: {code}, message: {message}".format(
                            code=error["code"], message=error['message']
                        )
                    )
                elif "stream" in line_payload:
                    logger.info(line_payload['stream'])


def add_local_bentoml_package_to_repo(archive_path):
    try:
        module_location, = importlib.util.find_spec(
            'bentoml'
        ).submodule_search_locations
    except AttributeError:
        # python 2.7 doesn't have importlib.util, will fall back to imp instead
        import imp

        _, module_location, _ = imp.find_module('bentoml')
    bentoml_setup_py = os.path.abspath(os.path.join(module_location, '..', 'setup.py'))

    assert os.path.isfile(bentoml_setup_py), '"setup.py" for Bentoml module not found'

    # Create random directory inside bentoml module for storing the bundled
    # targz file. Since dist-dir can only be inside of the module directory
    bundle_dir_name = '__bento_dev_tmp'
    Path(bundle_dir_name).mkdir(exist_ok=True, parents=True)

    sandbox.run_setup(
        bentoml_setup_py, ['sdist', '--format', 'gztar', '--dist-dir', bundle_dir_name]
    )

    # copy the generated targz to archive directory and remove it from
    # bentoml module directory
    source_dir = os.path.abspath(os.path.join(module_location, '..', bundle_dir_name))
    dest_dir = os.path.join(archive_path, 'bundled_dependencies')
    shutil.copytree(source_dir, dest_dir)
    shutil.rmtree(source_dir)

    return
