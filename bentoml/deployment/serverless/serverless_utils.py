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

from ruamel.yaml import YAML

from bentoml.configuration import BENTOML_HOME
from bentoml.utils import Path
from bentoml.utils.tempdir import TempDirectory
from bentoml.utils.whichcraft import which
from bentoml.exceptions import BentoMLException, BentoMLMissingDepdencyException

logger = logging.getLogger(__name__)


SERVERLESS_VERSION = '1.53.0'

# We will install serverless package and use the installed one, instead
# of user's installation
SERVERLESS_BIN_COMMAND = '{}/node_modules/.bin/serverless'.format(BENTOML_HOME)


def install_serverless_package():
    if not os.path.isfile(SERVERLESS_BIN_COMMAND):
        if which('npm') is None:
            raise BentoMLMissingDepdencyException(
                'Node and NPM is not installed.'
                ' Please visit www.nodejs.org for instructions'
            )
        install_command = ['npm', 'install', 'serverless@{}'.format(SERVERLESS_VERSION)]
        subprocess.call(install_command, cwd=BENTOML_HOME)


def install_serverless_plugin(plugin_name, install_dir_path):
    command = ["plugin", "install", "-n", plugin_name]
    call_serverless_command(command, install_dir_path)


def call_serverless_command(command, cwd_path):
    command = [SERVERLESS_BIN_COMMAND] + command

    with subprocess.Popen(command, cwd=cwd_path, stdout=PIPE, stderr=PIPE) as proc:
        response = parse_serverless_response(proc.stdout.read().decode("utf-8"))
        logger.debug("Serverless response: %s", "\n".join(response))
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


class TemporaryServerlessContent(object):
    def __init__(
        self, archive_path, deployment_name, bento_name, template_type, _cleanup=True
    ):
        self.archive_path = archive_path
        self.deployment_name = deployment_name
        self.bento_name = bento_name
        self.temp_directory = TempDirectory()
        self.template_type = template_type
        self._cleanup = _cleanup
        self.path = None

    def __enter__(self):
        self.generate()
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cleanup:
            self.cleanup()

    def generate(self):
        install_serverless_package()
        self.temp_directory.create()
        tempdir = self.temp_directory.path
        call_serverless_command(
            [
                "create",
                "--template",
                self.template_type,
                "--name",
                self.deployment_name,
            ],
            tempdir,
        )
        requirement_txt_path = os.path.join(self.archive_path, 'requirements.txt')
        shutil.copy(requirement_txt_path, tempdir)
        bento_archive_path = os.path.join(tempdir, self.bento_name)
        model_path = os.path.join(self.archive_path, self.bento_name)
        shutil.copytree(model_path, bento_archive_path)

        bundled_dependencies_path = os.path.join(
            self.archive_path, 'bundled_pip_dependencies'
        )
        # If bundled_pip_dependencies directory exists, we copy over and update
        # requirements.txt.  We need to remove the bentoml entry in the file, because
        # when pip install, it will NOT override the pypi released version.
        if os.path.isdir(bundled_dependencies_path):
            dest_bundle_path = os.path.join(tempdir, 'bundled_pip_dependencies')
            shutil.copytree(bundled_dependencies_path, dest_bundle_path)
            bundled_files = os.listdir(dest_bundle_path)
            has_bentoml_bundle = False
            for index, bundled_file_name in enumerate(bundled_files):
                bundled_files[index] = './bundled_pip_dependencies/{}\n'.format(
                    bundled_file_name
                )
                # If file name start with `BentoML-`, assuming it is a
                # bentoml targz bundle
                if bundled_file_name.startswith('BentoML-'):
                    has_bentoml_bundle = True

            with open(
                os.path.join(tempdir, 'requirements.txt'), 'r+'
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

        self.path = tempdir

    def cleanup(self):
        self.temp_directory.cleanup()
        self.path = None


class TemporaryServerlessConfig(object):
    def __init__(
        self,
        archive_path,
        deployment_name,
        region,
        stage,
        functions,
        provider_name,
        _cleanup=True,
    ):
        self.archive_path = archive_path
        self.temp_directory = TempDirectory()
        self.deployment_name = deployment_name
        self.region = region
        self.stage = stage
        self.functions = functions
        self.provider_name = provider_name
        self._cleanup = _cleanup
        self.path = None

    def __enter__(self):
        self.generate()
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cleanup:
            self.cleanup()

    def generate(self):
        install_serverless_package()
        serverless_config = {
            "service": self.deployment_name,
            "provider": {
                "region": self.region,
                "stage": self.stage,
                "name": self.provider_name,
            },
            "functions": self.functions,
        }

        # if self.platform == "google-python":
        #     serverless_config["provider"]["name"] = "google"
        #     for api in apis:
        #         serverless_config["functions"][api.name] = {
        #             "handler": api.name,
        #             "events": [{"http": "path"}],
        #         }

        yaml = YAML()
        self.temp_directory.create()
        tempdir = self.temp_directory.path
        saved_path = os.path.join(tempdir, "serverless.yml")
        yaml.dump(serverless_config, Path(saved_path))
        self.path = tempdir

    def cleanup(self):
        self.temp_directory.cleanup()
        self.path = None
