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
from bentoml.exceptions import BentoMLDeploymentException, BentoMLException
from bentoml.proto.deployment_pb2 import (
    Deployment,
    ApplyDeploymentResponse,
    DescribeDeploymentResponse,
    DeleteDeploymentResponse,
    DeploymentState,
)
from bentoml.deployment.serverless.serverless_utils import (
    call_serverless_command,
    generate_bundle,
    create_temporary_yaml_config,
)
from bentoml.archive.loader import load_bentoml_config

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
            "handler": "handler.{name}".format(name=api['name']),
            "events": [
                {"http": {"path": "/{name}".format(name=api['name']), "method": "post"}}
            ],
        }
        serverless_config["functions"][api['name']] = function_config

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
        api_content = AWS_FUNCTION_TEMPLATE.format(api_name=api['name'])
        handler_py_content = handler_py_content + api_content

    with open(os.path.join(output_path, "handler.py"), "w") as f:
        f.write(handler_py_content)
    return


def update_additional_lambda_config(dir_path, bento_config, region, stage):
    generate_handler_py(bento_archive.name, bento_config['apis'], dir_path)
    generate_serverless_configuration_for_aws_lambda(
        bento_archive.name, bento_config['apis'], dir_path, region, stage
    )
    return


def generate_temp_serverless_config_for_aws_lambda(bento_archive, region, stage):
    functions = {}
    for api in bento_archive.apis:
        functions[api.name] = {
            "handler": "handler." + api['name'],
            "events": [{"http": {"path": "/" + api['name'], "method": "post"}}],
        }

    return create_temporary_yaml_config(
        'aws', region, stage, bento_archive.name, functions
    )


class AwsLambdaDeploymentOperator(DeploymentOperatorBase):
    def apply(self, deployment_pb, repo=None):
        deployment_spec = deployment_pb.spec

        bento_path = repo.get(deployment_spec.bento_name, deployment_spec.bento_version)
        output_path = generate_bundle(bento_path, deployment_spec.bento_name)
        bento_config = load_bentoml_config(bento_path)

        update_additional_lambda_config(
            output_path,
            bento_config,
            deployment_spec.aws_lambda_operator_config.region,
            deployment_spec.aws_lambda_operator_config.stage,
        )

        call_serverless_command(["serverless", "deploy"], output_path)

        res_deployment_pb = Deployment()
        res_deployment_pb.CopyFrom(deployment_pb)
        state = self.describe(res_deployment_pb).state
        res_deployment_pb.state = state
        return ApplyDeploymentResponse(status=Status.OK(), deployment=res_deployment_pb)

    def delete(self, deployment_pb, repo=None):
        deployment = self.get(deployment_pb).deployment
        if deployment.state.state != DeploymentState.RUNNING:
            raise BentoMLDeploymentException(
                "No active deployment: %s" % deployment.name
            )

        deployment_spec = deployment_pb.spec

        bento_path = repo.get(deployment_spec.bento_name, deployment_spec.bento_version)
        tempdir = generate_temp_serverless_config_for_aws_lambda(
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

    def describe(self, deployment_pb, repo=None):
        deployment_spec = deployment_pb.spec

        bento_path = repo.get(deployment_spec.bento_name, deployment_spec.bento_version)
        tempdir = generate_temp_serverless_config_for_aws_lambda(
            bento_archive,
            deployment_spec.aws_lambda_operator_config.region,
            deployment_spec.aws_lambda_operator_config.stage,
        )

        try:
            response = call_serverless_command(["serverless", "info"], tempdir)
            state = DeploymentState(
                state=DeploymentState.RUNNING, info_json="\n".join(response)
            )
        except BentoMLException as e:
            state = DeploymentState(state=DeploymentState.ERROR, error_message=str(e))

        return DescribeDeploymentResponse(status=Status.OK(), state=state)
