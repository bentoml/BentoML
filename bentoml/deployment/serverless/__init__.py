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
from bentoml.deployment.base_deployment import Deployment
from bentoml.deployment.serverless.aws_lambda_template import (
    create_aws_lambda_bundle,
    DEFAULT_AWS_DEPLOY_STAGE,
    DEFAULT_AWS_REGION,
)
from bentoml.deployment.serverless.gcp_function_template import (
    create_gcp_function_bundle,
    DEFAULT_GCP_REGION,
    DEFAULT_GCP_DEPLOY_STAGE,
)
from bentoml.deployment.utils import generate_bentoml_deployment_snapshot_path

logger = logging.getLogger(__name__)

SERVERLESS_PROVIDER = {
    "aws-lambda": "aws-python3",
    "aws-lambda-py2": "aws-python",
    "gcp-function": "google-python",
}


def check_serverless_compatiable_version():
    if which("serverless") is None:
        raise ValueError(
            "Serverless framework is not installed, please visit "
            + "www.serverless.com for install instructions."
        )

    version_result = subprocess.check_output(["serverless", "-v"])
    parsed_version = version.parse(version_result.decode("utf-8").strip())

    if parsed_version >= version.parse("1.40.0"):
        return
    else:
        raise ValueError(
            "Incompatiable serverless version, please install version 1.40.0 or greater"
        )


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


class ServerlessDeployment(Deployment):
    """Managing deployment operations for serverless
    """

    def __init__(self, archive_path, platform, region, stage):
        check_serverless_compatiable_version()
        super(ServerlessDeployment, self).__init__(archive_path)

        self.platform = platform
        self.provider = SERVERLESS_PROVIDER[platform]
        if platform == "google-python":
            self.region = DEFAULT_GCP_REGION if region is None else region
            self.stage = DEFAULT_GCP_DEPLOY_STAGE if stage is None else stage
        elif platform == "aws-lambda" or platform == "aws-lambda-py":
            self.region = DEFAULT_AWS_REGION if region is None else region
            self.stage = DEFAULT_AWS_DEPLOY_STAGE if stage is None else stage
        else:
            raise ValueError(
                "This version of BentoML doesn't support platform %s" % platform
            )

    def _generate_bundle(self):
        output_path = generate_bentoml_deployment_snapshot_path(
            self.bento_service.name, self.bento_service.version, self.platform
        )
        Path(output_path).mkdir(parents=True, exist_ok=False)

        # Calling serverless command to generate templated project
        with subprocess.Popen(
            [
                "serverless",
                "create",
                "--template",
                self.provider,
                "--name",
                self.bento_service.name,
            ],
            cwd=output_path,
            stdout=PIPE,
            stderr=PIPE,
        ) as proc:
            response = parse_serverless_response(proc.stdout.read().decode("utf-8"))
            logger.debug("Serverless response: %s", "\n".join(response))

        if self.platform == "google-python":
            create_gcp_function_bundle(
                self.bento_service, output_path, self.region, self.stage
            )
        elif self.platform == "aws-lambda" or self.platform == "aws-lambda-py2":
            # Installing two additional plugins to make it works for AWS lambda
            # serverless-python-requirements will packaging required python modules,
            # and automatically compress and create layer
            with subprocess.Popen(
                [
                    "serverless",
                    "plugin",
                    "install",
                    "-n",
                    "serverless-python-requirements",
                ],
                cwd=output_path,
                stdout=PIPE,
                stderr=PIPE,
            ) as proc:
                response = parse_serverless_response(proc.stdout.read().decode("utf-8"))
                logger.debug("Serverless response: %s", "\n".join(response))

            with subprocess.Popen(
                ["serverless", "plugin", "install", "-n", "serverless-apigw-binary"],
                cwd=output_path,
                stdout=PIPE,
                stderr=PIPE,
            ) as proc:
                response = parse_serverless_response(proc.stdout.read().decode("utf-8"))
                logger.debug("Serverless response: %s", "\n".join(response))

            create_aws_lambda_bundle(
                self.bento_service, output_path, self.region, self.stage
            )
        else:
            raise BentoMLException(
                "%s is not supported in current version of BentoML" % self.provider
            )

        shutil.copy(os.path.join(self.archive_path, "requirements.txt"), output_path)

        model_serivce_archive_path = os.path.join(output_path, self.bento_service.name)
        shutil.copytree(self.archive_path, model_serivce_archive_path)

        return os.path.realpath(output_path)

    def deploy(self):
        output_path = self._generate_bundle()
        with subprocess.Popen(
            ["serverless", "deploy"], cwd=output_path, stdout=PIPE, stderr=PIPE
        ) as proc:
            response = parse_serverless_response(proc.stdout.read().decode("utf-8"))
            logger.debug("Serverless response: %s", "\n".join(response))
            service_info_index = response.index("Service Information")
            service_info = response[service_info_index:]
            logger.info("BentoML: %s", "\n".join(service_info))
            print("\n".join(service_info))
            return output_path

    def check_status(self):
        """Check deployment status for the bentoml service.
        return True, if it is active else return false
        """

        apis = self.bento_service.get_service_apis()
        config = {
            "service": self.bento_service.name,
            "provider": {"region": self.region, "stage": self.stage},
            "functions": {},
        }
        if self.platform == "google-python":
            config["provider"]["name"] = "google"
            for api in apis:
                config["functions"][api.name] = {
                    "handler": api.name,
                    "events": [{"http": "path"}],
                }
        elif self.platform == "aws-lambda" or self.platform == "aws-lambda-py2":
            config["provider"]["name"] = "aws"
            for api in apis:
                config["functions"][api.name] = {
                    "handler": "handler." + api.name,
                    "events": [{"http": {"path": "/" + api.name, "method": "post"}}],
                }
        else:
            raise BentoMLException(
                "check serverless does not support platform %s at the moment"
                % self.platform
            )
        yaml = YAML()

        with TempDirectory() as tempdir:
            saved_path = os.path.join(tempdir, "serverless.yml")
            yaml.dump(config, Path(saved_path))
            with subprocess.Popen(
                ["serverless", "info"], cwd=tempdir, stdout=PIPE, stderr=PIPE
            ) as proc:
                # We don't use the parse_response function here.
                # Instead of raising error, we will just return false
                content = proc.stdout.read().decode("utf-8")
                response = content.strip().split("\n")
                logger.debug("Serverless response: %s", "\n".join(response))
                error = [s for s in response if "Serverless Error" in s]
                if error:
                    print("has error", "\n".join(response))
                    return False, "\n".join(response)
                else:
                    print("\n".join(response))
                    return True, "\n".join(response)

    def delete(self):
        is_active, _ = self.check_status()
        if not is_active:
            raise BentoMLException(
                "No active deployment for service %s" % self.bento_service.name
            )

        if self.platform == "google-python":
            provider_name = "google"
        elif self.platform == "aws-lambda" or self.platform == "aws-lambda-py2":
            provider_name = "aws"
        config = {
            "service": self.bento_service.name,
            "provider": {
                "name": provider_name,
                "region": self.region,
                "stage": self.stage,
            },
        }
        yaml = YAML()
        with TempDirectory() as tempdir:
            saved_path = os.path.join(tempdir, "serverless.yml")
            yaml.dump(config, Path(saved_path))

            with subprocess.Popen(
                ["serverless", "remove"], cwd=tempdir, stdout=PIPE, stderr=PIPE
            ) as proc:
                response = parse_serverless_response(proc.stdout.read().decode("utf-8"))
                logger.debug("Serverless response: %s", "\n".join(response))
                if self.platform == "google-python":
                    # TODO: Add check for Google's response
                    return True
                elif self.platform == "aws-lambda" or self.platform == "aws-lambda-py2":
                    if "Serverless: Stack removal finished..." in response:
                        return True
                    else:
                        return False
