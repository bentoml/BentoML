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
from packaging import version

from bentoml.utils import Path
from bentoml.utils.tempdir import TempDirectory
from bentoml.utils.whichcraft import which
from bentoml.exceptions import BentoMLException

logger = logging.getLogger(__name__)

MINIMUM_SERVERLESS_VERSION = '1.40.0'


def check_serverless_compatiable_version():
    if which("serverless") is None:
        raise ValueError(
            "Serverless framework is not installed, please visit "
            + "www.serverless.com for install instructions."
        )

    version_result = (
        subprocess.check_output(["serverless", "-v"]).decode("utf-8").strip()
    )
    if "(Enterprise Plugin:" in version_result:
        slice_end_index = version_result.find(" (Enterprise")
        version_result = version_result[0:slice_end_index]
    parsed_version = version.parse(version_result)

    if parsed_version >= version.parse(MINIMUM_SERVERLESS_VERSION):
        return
    else:
        raise ValueError(
            "Incompatiable serverless version, please install version 1.40.0 or greater"
        )


def install_serverless_plugin(plugin_name, install_dir_path):
    command = ["serverless", "plugin", "install", "-n", plugin_name]
    call_serverless_command(command, install_dir_path)


def call_serverless_command(command, cwd_path):
    with subprocess.Popen(command, cwd=cwd_path, stdout=PIPE, stderr=PIPE) as proc:
        response = parse_serverless_response(proc.stdout.read().decode("utf-8"))
        logger.debug("Serverless response: %s", "\n".join(response))
    return response


def parse_serverless_response(serverless_response):
    """Parse serverless response string, raise error if it is a serverless error,
  otherwise, return information.
  """
    str_list = serverless_response.strip().split("\n")
    error = [s for s in str_list if "Serverless Error" in s]
    if error:
        error_pos = str_list.index(error[0])
        error_message = str_list[error_pos + 2]
        raise BentoMLException(error_message)
    return str_list


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
        self.temp_directory.create()
        tempdir = self.temp_directory.path
        call_serverless_command(
            [
                "serverless",
                "create",
                "--template",
                self.template_type,
                "--name",
                self.deployment_name,
            ],
            tempdir,
        )
        shutil.copy(os.path.join(self.archive_path, "requirements.txt"), tempdir)
        model_serivce_archive_path = os.path.join(tempdir, self.bento_name)
        model_path = os.path.join(self.archive_path, self.bento_name)
        shutil.copytree(model_path, model_serivce_archive_path)
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
