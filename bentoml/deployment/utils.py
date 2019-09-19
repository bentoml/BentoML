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
import imp

from setuptools import sandbox


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


INSTALL_WHEEL_TEMPLATE = """\
#!/bin/bash

for filename in ./wheel_dependencies/*.whl; do
    [ -e "$filename" ] || continue
    # pip install "$filename" --ignore-installed
done
"""


def add_local_bentoml_package_to_repo(deployment_pb, repo):
    deployment_spec = deployment_pb.spec
    archive_path = repo.get(deployment_spec.bento_name, deployment_spec.bento_version)
    bentoml_location = imp.find_module('bentoml')[1]

    wheel_dir = os.path.join(archive_path, 'wheel_dependencies')
    install_script_path = os.path.join(archive_path, 'install_wheels.sh')
    setup_py = os.path.join(bentoml_location, 'setup.py')
    sandbox.run_setup(setup_py, ['bdist_wheel', '--dist-dir', wheel_dir])

    with open(install_script_path, 'w') as f:
        f.write(INSTALL_WHEEL_TEMPLATE)
    permission = "755"
    octal_permission = int(permission, 8)
    os.chmod(install_script_path, octal_permission)

    return
