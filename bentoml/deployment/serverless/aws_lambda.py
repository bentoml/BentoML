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
from packaging import version

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
    install_serverless_plugin,
    TemporaryServerlessContent,
    TemporaryServerlessConfig,
    parse_serverless_info_response_to_json_string,
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


def generate_aws_handler_functions_config(apis):
    function_list = {}
    for api in apis:
        function_list[api['name']] = {
            "handler": "handler." + api['name'],
            "events": [{"http": {"path": "/" + api['name'], "method": "post"}}],
        }
    return function_list


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

    serverless_config["functions"] = generate_aws_handler_functions_config(apis)

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


class AwsLambdaDeploymentOperator(DeploymentOperatorBase):
    def apply(self, deployment_pb, repo=None):
        deployment_spec = deployment_pb.spec
        aws_config = deployment_spec.aws_lambda_operator_config

        bento_path = repo.get(deployment_spec.bento_name, deployment_spec.bento_version)
        bento_config = load_bentoml_config(bento_path)

        template = 'aws-python3'
        test_version = version.parse('3.0.0')
        package_version = version.parse(bento_config['env']['python_version'])
        if package_version > test_version:
            template = 'aws-python'

        with TemporaryServerlessContent(
            archive_path=bento_path,
            deployment_name=deployment_pb.name,
            bento_name=deployment_spec.bento_name,
            template_type=template,
        ) as output_path:
            generate_handler_py(
                deployment_spec.bento_name, bento_config['apis'], output_path
            )
            generate_serverless_configuration_for_aws_lambda(
                service_name=deployment_pb.name,
                apis=bento_config['apis'],
                output_path=output_path,
                region=aws_config.region,
                stage=deployment_pb.namespace + '-' + aws_config.stage,
            )
            try:
                logger.info(
                    'Installing additional packages: serverless-python-requirements, serverless-apigw-binary'
                )
                install_serverless_plugin("serverless-python-requirements", output_path)
                install_serverless_plugin("serverless-apigw-binary", output_path)
                logger.info('Deploying to AWS Lambda')
                call_serverless_command(["serverless", "deploy"], output_path)
            except BentoMLException as e:
                raise BentoMLDeploymentException(
                    'Failed to deploy with AWS Lambda for deployment %s.',
                    deployment_pb.name,
                )

        res_deployment_pb = Deployment(state=DeploymentState())
        res_deployment_pb.CopyFrom(deployment_pb)
        state = self.describe(res_deployment_pb, repo).state
        res_deployment_pb.state.CopyFrom(state)
        return ApplyDeploymentResponse(status=Status.OK(), deployment=res_deployment_pb)

    def delete(self, deployment_pb, repo=None):
        state = self.describe(deployment_pb, repo).state
        if state.state != DeploymentState.RUNNING:
            raise BentoMLDeploymentException(
                "No active deployment: %s" % deployment_pb.name
            )

        deployment_spec = deployment_pb.spec
        aws_config = deployment_spec.aws_lambda_operator_config

        bento_path = repo.get(deployment_spec.bento_name, deployment_spec.bento_version)
        bento_config = load_bentoml_config(bento_path)

        with TemporaryServerlessConfig(
            archive_path=bento_path,
            deployment_name=deployment_pb.name,
            region=aws_config.region,
            stage=deployment_pb.namespace + '-' + aws_config.stage,
            provider_name='aws',
            functions=generate_aws_handler_functions_config(bento_config['apis']),
        ) as tempdir:
            response = call_serverless_command(['serverless', 'remove'], tempdir)
            if "Serverless: Stack removal finished..." in response:
                status = Status.OK()
            else:
                status = Status.ABORTED()

        return DeleteDeploymentResponse(status=status)

    def describe(self, deployment_pb, repo=None):
        deployment_spec = deployment_pb.spec
        aws_config = deployment_spec.aws_lambda_operator_config

        # bento_path = repo.get(deployment_spec.bento_name, deployment_spec.bento_version)
        bento_path = repo
        bento_config = load_bentoml_config(bento_path)
        with TemporaryServerlessConfig(
            archive_path=bento_path,
            deployment_name=deployment_pb.name,
            region=aws_config.region,
            stage=deployment_pb.namespace + '-' + aws_config.stage,
            provider_name='aws',
            functions=generate_aws_handler_functions_config(bento_config['apis']),
        ) as tempdir:
            try:
                response = call_serverless_command(["serverless", "info"], tempdir)
                info_json = parse_serverless_info_response_to_json_string(response)
                state = DeploymentState(
                    state=DeploymentState.RUNNING, info_json=info_json
                )
            except BentoMLException as e:
                state = DeploymentState(
                    state=DeploymentState.ERROR, error_message=str(e)
                )

        return DescribeDeploymentResponse(status=Status.OK(), state=state)
