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
from subprocess import PIPE

from ruamel.yaml import YAML
from packaging import version

from bentoml.utils import Path
from bentoml.utils.tempdir import TempDirectory
from bentoml.utils.whichcraft import which
from bentoml.exceptions import BentoMLException

logger = logging.getLogger(__name__)


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

    if parsed_version >= version.parse("1.40.0"):
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
        error_message = str_list[error_pos + 1]
        raise BentoMLException(error_message)
    return str_list


def generate_bundle(archive_path, template_type, bento_name):
    with TempDirectory() as tempdir:
        # Calling serverless command to generate templated project
        call_serverless_command(
            ["serverless", "create", "--template", template_type, "--name", bento_name],
            tempdir,
        )

        # if self.platform == "google-python":
        #     create_gcp_function_bundle(
        #       self.bento_service, output_path, self.region, self.stage
        #     )
        # elif self.platform == "aws-lambda" or self.platform == "aws-lambda-py2":
        #     # Installing two additional plugins to make it works for AWS lambda
        #     # serverless-python-requirements will packaging required python modules,
        #     # and automatically compress and create layer
        #     install_serverless_plugin("serverless-python-requirements", output_path)
        #     install_serverless_plugin("serverless-apigw-binary", output_path)
        #
        #     create_aws_lambda_bundle(
        #       self.bento_service, output_path, self.region, self.stage
        #     )
        # else:
        #     raise BentoMLException(
        #       "%s is not supported in current version of BentoML" % self.provider
        #     )

        shutil.copy(os.path.join(archive_path, "requirements.txt"), tempdir)
        model_serivce_archive_path = os.path.join(tempdir, bento_name)
        shutil.copytree(archive_path, model_serivce_archive_path)

    return os.path.realpath(tempdir)


def create_temporary_yaml_config(provider_name, region, stage, bento_name, functions):
    serverless_config = {
        "service": bento_name,
        "provider": {"region": region, "stage": stage, "name": provider_name},
        "functions": functions,
    }

    # if self.platform == "google-python":
    #     serverless_config["provider"]["name"] = "google"
    #     for api in apis:
    #         serverless_config["functions"][api.name] = {
    #             "handler": api.name,
    #             "events": [{"http": "path"}],
    #         }

    yaml = YAML()
    with TempDirectory() as tempdir:
        saved_path = os.path.join(tempdir, "serverless.yml")
        yaml.dump(serverless_config, Path(saved_path))
    return tempdir
