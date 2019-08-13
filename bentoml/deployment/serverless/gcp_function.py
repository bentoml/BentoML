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
import logging

from ruamel.yaml import YAML

from bentoml.utils import Path
from bentoml.deployment.operator import DeploymentOperatorBase
from bentoml.yatai.status import Status
from bentoml.repository import get_local
from bentoml.exceptions import BentoMLDeploymentException, BentoMLException
from bentoml.proto.deployment_pb2 import (
    ApplyDeploymentResponse,
    GetDeploymentResponse,
    DeleteDeploymentResponse,
    DeploymentState,
)
from bentoml.deployment.serverless.serverless_utils import (
    call_serverless_command,
    generate_bundle,
    create_temporary_yaml_config,
)

logger = logging.getLogger(__name__)

GOOGLE_MAIN_PY_TEMPLATE_HEADER = """\
from {class_name} import {class_name}

bento_service = {class_name}.load()

"""

GOOGLE_FUNCTION_TEMPLATE = """\
def {api_name}(request):
    api = bento_service.get_service_api('{api_name}')

    return api.handle_request(request)

"""

def generate_serverless_configuration_for_gcp_function(
    bento_service, apis, output_path, region, stage
):
    config_path = os.path.join(output_path, "serverless.yml")
    yaml = YAML()
    with open(config_path, "r") as f:
        content = f.read()
    serverless_config = yaml.load(content)

    serverless_config["service"] = bento_service.name
    serverless_config["provider"]["project"] = bento_service.name

    serverless_config["provider"]["region"] = region
    logger.info("Using user defined Google region: %s", region)

    serverless_config["provider"]["stage"] = stage
    logger.info("Using user defined Google stage: %s", stage)

    serverless_config["functions"] = {}
    for api in apis:
        function_config = {"handler": api.name, "events": [{"http": "path"}]}
        serverless_config["functions"][api.name] = function_config

    yaml.dump(serverless_config, Path(config_path))
    return


def generate_main_py(bento_service, apis, output_path):
    handler_py_content = GOOGLE_MAIN_PY_TEMPLATE_HEADER.format(
        class_name=bento_service.name
    )

    for api in apis:
        api_content = GOOGLE_FUNCTION_TEMPLATE.format(api_name=api.name)
        handler_py_content = handler_py_content + api_content

    with open(os.path.join(output_path, "main.py"), "w") as f:
        f.write(handler_py_content)
    return


def generate_temp_serverless_config_for_gcp_function(bento_archive, region, stage):
    functions = {}
    for api in bento_archive.apis:
        functions[api.name] = {
            "handler": api.name,
            "events": [{"http": "path"}],
        }

    return create_temporary_yaml_config(
        'google', region, stage, bento_archive.name, functions
    )

def generate_handler_py(bento_name, apis, output_path):
    handler_py_content = GOOGLE_MAIN_PY_TEMPLATE_HEADER.format(class_name=bento_name)

    for api in apis:
        api_content = GOOGLE_FUNCTION_TEMPLATE.format(api_name=api.name)
        handler_py_content = handler_py_content + api_content

    with open(os.path.join(output_path, "handler.py"), "w") as f:
        f.write(handler_py_content)
    return

def update_additional_lambda_config(dir_path, bento_archive, region, stage):
    generate_handler_py(bento_archive.name, bento_archive.apis, dir_path)
    generate_serverless_configuration_for_gcp_function(
        bento_archive.name, bento_archive.apis, dir_path, region, stage
    )
    return

class GcpFunctionDeploymentOperator(DeploymentOperatorBase):
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

        call_serverless_command(
            ["serverless", "deploy"], output_path
        )

        deployment = self.get(deployment_pb).deployment
        return ApplyDeploymentResponse(status=Status.OK(), deployment=deployment)


    def get(self, deployment_pb):
        deployment_spec = deployment_pb.spec

        repository = get_local()
        bento_archive = repository.get(
            deployment_spec.bento_name, deployment_spec.bento_version
        )
        tempdir = generate_temp_serverless_config_for_gcp_function(
            bento_archive,
            deployment_spec.aws_lambda_operator_config.region,
            deployment_spec.aws_lambda_operator_config.stage,
        )

        try:
            response = call_serverless_command(["serverless", "info"], tempdir)
            deployment_pb.state = DeploymentState(
                state=DeploymentState.RUNNING, info_json="\n".join(response)
            )
        except BentoMLException as e:
            deployment_pb.state = DeploymentState(
                state=DeploymentState.ERROR, error_message=str(e)
            )

        return GetDeploymentResponse(status=Status.OK(), deployment=deployment_pb)

    def delete(self, deployment_pb):
        deployment = self.get(deployment_pb).deployment
        if deployment.state.state != DeploymentState.RUNNING:
            raise BentoMLDeploymentException(
                "No active deployment: %s" % deployment.name
            )

        deployment_spec = deployment_pb.spec

        repository = get_local()
        bento_archive = repository.get(
            deployment_spec.bento_name, deployment_spec.bento_version
        )
        tempdir = generate_temp_serverless_config_for_gcp_function(
            bento_archive,
            deployment_spec.aws_lambda_operator_config.region,
            deployment_spec.aws_lambda_operator_config.stage,
        )

        response = call_serverless_command(['serverless', 'remove'], tempdir)

        if "Serverless: Stack removal finished..." in response:
            status = Status.OK()
        else:
            status = Status.ABORTED()
        return DeleteDeploymentResponse(status=status)

