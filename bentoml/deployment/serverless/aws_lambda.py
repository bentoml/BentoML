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

import logging

from bentoml.deployment.operator import DeploymentOperatorBase
from bentoml.yatai.status import Status
from bentoml.repository import get_local
from bentoml.exceptions import BentoMLDeploymentException
from bentoml.proto.deployment_pb2 import (
    ApplyDeploymentResponse,
    GetDeploymentResponse,
    DeleteDeploymentResponse,
)
from bentoml.deployment.serverless.serverless_utils import (
    call_serverless_command,
    generate_bundle,
    create_temporary_yaml_config,
)

logger = logging.getLogger(__name__)

AWS_HANDLER_PY_TEMPLATE_HEADER = """\
try:
    import unzip_requirements
except ImportError:
    pass

from {class_name} import {class_name}

bento_service = {class_name}.load()

"""

AWS_FUNCTION_TEMPLATE = """\
def {api_name}(event, context):
    api = bento_service.get_service_api('{api_name}')

    return api.handle_aws_lambda_event(event)

"""


def generate_serverless_configuration_for_aws_lambda(
    service_name, apis, output_path, region, stage
):
    config_path = os.path.join(output_path, "serverless.yml")
    yaml = YAML()
    with open(config_path, "r") as f:
        content = f.read()
    serverless_config = yaml.load(content)

    serverless_config["service"] = service_name
    serverless_config["provider"]["region"] = region
    logger.info("Using user AWS region: %s", region)

    serverless_config["provider"]["stage"] = stage
    logger.info("Using AWS stage: %s", stage)

    serverless_config["functions"] = {}
    for api in apis:
        function_config = {
            "handler": "handler.{name}".format(name=api.name),
            "events": [
                {"http": {"path": "/{name}".format(name=api.name), "method": "post"}}
            ],
        }
        serverless_config["functions"][api.name] = function_config

    serverless_config["custom"] = {
        "apigwBinary": ["image/jpg", "image/jpeg", "image/png"],
        "pythonRequirements": {
            "useDownloadCache": True,
            "useStaticCache": True,
            "dockerizePip": True,
            "slim": True,
            "strip": True,
            "zip": True,
        },
    }

    yaml.dump(serverless_config, Path(config_path))
    return


def generate_handler_py(bento_name, apis, output_path):
    handler_py_content = AWS_HANDLER_PY_TEMPLATE_HEADER.format(class_name=bento_name)

    for api in apis:
        api_content = AWS_FUNCTION_TEMPLATE.format(api_name=api.name)
        handler_py_content = handler_py_content + api_content

    with open(os.path.join(output_path, "handler.py"), "w") as f:
        f.write(handler_py_content)
    return


def update_additional_lambda_config(dir_path, bento_archive, region, stage):
    generate_handler_py(bento_archive.name, bento_archive.apis, dir_path)
    generate_serverless_configuration_for_aws_lambda(
        bento_archive.name, bento_archive.apis, dir_path, region, stage
    )
    return


class AwsLambdaDeploymentOperator(DeploymentOperatorBase):
    def apply(self, deployment_pb):
        deployment_spec = deployment_pb.spec

        repository = get_local()
        bento_archive = repository.get(
            deployment_spec.bento_name, deployment_spec.bento_version
        )
        archive_path = bento_archive.uri.uri
        output_path = generate_bundle(archive_path, bento_archive.name)

        update_additional_lambda_config(
            output_path,
            bento_archive,
            deployment_spec.aws_lambda_operator_config.region,
            deployment_spec.aws_lambda_operator_config.stage,
        )

        def display_deployed_info(response):
            service_info_index = response.index("Service Information")
            service_info = response[service_info_index:]
            logger.info("BentoML: %s", "\n".join(service_info))
            print("\n".join(service_info))

        call_serverless_command(
            ["serverless", "deploy"], output_path, display_deployed_info
        )

        deployment = self.get(deployment_pb).deployment
        return ApplyDeploymentResponse(status=Status.OK, deployment=deployment)

    def get(self, deployment_pb):

        tempdir = create_temporary_yaml_config()

        def parse_status_response(response):
            error = [s for s in response if "Serverless Error" in s]
            if error:
                print("has error", "\n".join(response))
                return False, "\n".join(response)
            else:
                print("\n".join(response))
                return True, "\n".join(response)

        return call_serverless_command(
            ["serverless", "info"], tempdir, parse_status_response
        )

        return GetDeploymentResponse()

    def delete(self, deployment_pb):
        # deployment = self.get(deployment_pb).deployment

        is_active, _ = self.check_status()
        if not is_active:
            raise BentoMLDeploymentException(
                "No active deployment for service %s" % self.bento_service.name
            )
        tempdir = self._create_temporary_yaml_config()

        def parse_deletion_response(response):
            if self.platform == "google-python":
                # TODO: Add check for Google's response
                return True
            elif self.platform == "aws-lambda" or self.platform == "aws-lambda-py2":
                if "Serverless: Stack removal finished..." in response:
                    return True
                else:
                    return False

        call_serverless_command(
            ['serverless', 'remove'], tempdir, parse_deletion_response
        )

        return DeleteDeploymentResponse()
