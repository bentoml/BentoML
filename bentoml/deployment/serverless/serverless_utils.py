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
import logging
import subprocess
import json
from subprocess import PIPE
from packaging import version


from bentoml.configuration import _get_bentoml_home
from bentoml.exceptions import BentoMLException, BentoMLMissingDependencyException

logger = logging.getLogger(__name__)


SERVERLESS_VERSION = '1.53.0'
BENTOML_HOME = _get_bentoml_home()

# We will install serverless package and use the installed one, instead
# of user's installation
SERVERLESS_BIN_COMMAND = '{}/node_modules/.bin/serverless'.format(BENTOML_HOME)


def check_nodejs_compatible_version():
    # We moved import which to inside this function, because it is easier to test with
    # with mock
    from bentoml.utils.whichcraft import which

    if which('npm') is None:
        raise BentoMLMissingDependencyException(
            'NPM is not installed. Please visit www.nodejs.org for instructions'
        )
    if which("node") is None:
        raise BentoMLMissingDependencyException(
            "NodeJs is not installed, please visit www.nodejs.org for install "
            "instructions."
        )
    version_result = subprocess.check_output(["node", "-v"]).decode("utf-8").strip()
    parsed_version = version.parse(version_result)

    if not parsed_version >= version.parse('v8.10.0'):
        raise ValueError(
            "Incompatible Nodejs version, please install version v8.10.0 " "or greater"
        )


def install_serverless_package():
    """ Install serverless npm package to BentoML home directory

    We are using serverless framework for deployment, instead of using user's own
    serverless framework, we will install a specific one just for BentoML.
    It will be installed in BentoML home directory.
    """
    check_nodejs_compatible_version()
    install_command = ['npm', 'install', 'serverless@{}'.format(SERVERLESS_VERSION)]
    try:
        subprocess.check_call(
            install_command, cwd=BENTOML_HOME, stdout=PIPE, stderr=PIPE
        )
    except subprocess.CalledProcessError as error:
        raise BentoMLException(error.output)


def install_serverless_plugin(plugin_name, install_dir_path):
    command = ["plugin", "install", "-n", plugin_name]
    call_serverless_command(command, install_dir_path)


def call_serverless_command(command, cwd_path):
    command = [SERVERLESS_BIN_COMMAND] + command

    with subprocess.Popen(command, cwd=cwd_path, stdout=PIPE, stderr=PIPE) as proc:
        stdout = proc.stdout.read().decode("utf-8")
        logger.debug("sls cmd output: %s", stdout)
        response = parse_serverless_response(stdout)
    return response


def parse_serverless_response(serverless_response):
    """Parse serverless response string, raise error if it is a serverless error,
  otherwise, return information.
  """
    serverless_outputs = serverless_response.strip().split("\n")

    # Parsing serverless response brutally.  The current serverless
    # response format is:
    # ServerlessError|Error -----{fill dash to 56 line length}
    # empty space
    # Error Message
    # empty space
    # We are just going to find the index of serverless error/error is
    # and raise Exception base on the message +2 index away from it.
    error_message = ''
    for index, message in enumerate(serverless_outputs):
        if 'Serverless Error' in message or 'Error -----' in message:
            error_message += serverless_outputs[index + 2]
    if error_message:
        raise BentoMLException(error_message)
    return serverless_outputs


def parse_serverless_info_response_to_json_string(responses):
    result = {}
    for i in range(len(responses)):
        line = responses[i]
        if ': ' in line:
            items = line.split(': ')
            result[items[0]] = items[1]
    result['raw_response'] = responses
    return json.dumps(result)


def init_serverless_project_dir(
    project_dir, archive_path, deployment_name, bento_name, template_type
):
    install_serverless_package()
    call_serverless_command(
        ["create", "--template", template_type, "--name", deployment_name], project_dir
    )
    requirement_txt_path = os.path.join(archive_path, 'requirements.txt')
    shutil.copy(requirement_txt_path, project_dir)
    bento_archive_path = os.path.join(project_dir, bento_name)
    model_path = os.path.join(archive_path, bento_name)
    shutil.copytree(model_path, bento_archive_path)

    bundled_dependencies_path = os.path.join(archive_path, 'bundled_pip_dependencies')
    # If bundled_pip_dependencies directory exists, we copy over and update
    # requirements.txt.  We need to remove the bentoml entry in the file, because
    # when pip install, it will NOT override the pypi released version.
    if os.path.isdir(bundled_dependencies_path):
        dest_bundle_path = os.path.join(project_dir, 'bundled_pip_dependencies')
        shutil.copytree(bundled_dependencies_path, dest_bundle_path)
        bundled_files = os.listdir(dest_bundle_path)
        has_bentoml_bundle = False
        for index, bundled_file_name in enumerate(bundled_files):
            bundled_files[index] = '\n./bundled_pip_dependencies/{}'.format(
                bundled_file_name
            )
            # If file name start with `BentoML-`, assuming it is a
            # bentoml targz bundle
            if bundled_file_name.startswith('BentoML-'):
                has_bentoml_bundle = True

        with open(
            os.path.join(project_dir, 'requirements.txt'), 'r+'
        ) as requirement_file:
            required_modules = requirement_file.readlines()
            if has_bentoml_bundle:
                # Assuming bentoml is always the first one in
                # requirements.txt. We are removing it
                required_modules = required_modules[1:]
            required_modules = required_modules + bundled_files
            # Write from beginning of the file, instead of appending to
            # the end.
            requirement_file.seek(0)
            requirement_file.writelines(required_modules)
