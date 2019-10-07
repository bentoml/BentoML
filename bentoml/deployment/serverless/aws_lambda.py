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
import json
from packaging import version

from ruamel.yaml import YAML
import boto3

from bentoml.exceptions import BentoMLException
from bentoml.utils import Path
from bentoml.deployment.operator import DeploymentOperatorBase
from bentoml.yatai.status import Status
from bentoml.proto.deployment_pb2 import (
    Deployment,
    ApplyDeploymentResponse,
    DescribeDeploymentResponse,
    DeleteDeploymentResponse,
    DeploymentState,
)
from bentoml.deployment.utils import (
    ensure_docker_available_or_raise,
    exception_to_return_status,
)
from bentoml.deployment.serverless.serverless_utils import (
    call_serverless_command,
    install_serverless_plugin,
    TemporaryServerlessContent,
    TemporaryServerlessConfig,
    generate_api_list,
)
from bentoml.archive.loader import load_bento_service_metadata

logger = logging.getLogger(__name__)

AWS_HANDLER_PY_TEMPLATE_HEADER = """\
import os
try:
    import unzip_requirements
except ImportError:
    pass

# Set BENTOML_HOME to /tmp directory due to AWS lambda disk access restrictions
os.environ['BENTOML_HOME'] = '/tmp/bentoml/'

from {class_name} import load

bento_service = load()

"""

AWS_FUNCTION_TEMPLATE = """\
def {api_name}(event, context):
    api = bento_service.get_service_api('{api_name}')

    return api.handle_aws_lambda_event(event)

"""


def generate_aws_handler_functions_config(apis):
    function_list = {}
    for api in apis:
        function_list[api.name] = {
            "handler": "handler." + api.name,
            "events": [{"http": {"path": "/" + api.name, "method": "post"}}],
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

    # We are passing the bundled_pip_dependencies directory for python
    # requirement package, so it can installs the bundled tar gz file.
    serverless_config["custom"] = {
        "apigwBinary": ["image/jpg", "image/jpeg", "image/png"],
        "pythonRequirements": {
            "useDownloadCache": True,
            "useStaticCache": True,
            "dockerizePip": True,
            "slim": True,
            "strip": True,
            "zip": True,
            "dockerRunCmdExtraArgs": [
                '-v',
                '{}/bundled_pip_dependencies:'
                '/var/task/bundled_pip_dependencies:z'.format(output_path),
            ],
        },
    }

    yaml.dump(serverless_config, Path(config_path))
    return


def generate_handler_py(bento_name, apis, output_path):
    with open(os.path.join(output_path, "handler.py"), "w") as f:
        f.write(AWS_HANDLER_PY_TEMPLATE_HEADER.format(class_name=bento_name))
        for api in apis:
            api_content = AWS_FUNCTION_TEMPLATE.format(api_name=api.name)
            f.write(api_content)
    return


class AwsLambdaDeploymentOperator(DeploymentOperatorBase):
    def apply(self, deployment_pb, repo, prev_deployment=None):
        try:
            ensure_docker_available_or_raise()
            deployment_spec = deployment_pb.spec
            aws_config = deployment_spec.aws_lambda_operator_config

            bento_path = repo.get(
                deployment_spec.bento_name, deployment_spec.bento_version
            )
            bento_config = load_bento_service_metadata(bento_path)

            template = 'aws-python3'
            minimum_python_version = version.parse('3.0.0')
            bento_python_version = version.parse(bento_config.env.python_version)
            if bento_python_version < minimum_python_version:
                template = 'aws-python'

            apis = generate_api_list(
                bento_config.apis, aws_config.api_name, deployment_spec.bento_name
            )

            with TemporaryServerlessContent(
                archive_path=bento_path,
                deployment_name=deployment_pb.name,
                bento_name=deployment_spec.bento_name,
                template_type=template,
            ) as output_path:
                generate_handler_py(deployment_spec.bento_name, apis, output_path)
                generate_serverless_configuration_for_aws_lambda(
                    service_name=deployment_pb.name,
                    apis=apis,
                    output_path=output_path,
                    region=aws_config.region,
                    stage=deployment_pb.namespace,
                )
                logger.info(
                    'Installing additional packages: serverless-python-requirements, '
                    'serverless-apigw-binary'
                )
                install_serverless_plugin("serverless-python-requirements", output_path)
                install_serverless_plugin("serverless-apigw-binary", output_path)
                logger.info('Deploying to AWS Lambda')
                call_serverless_command(["deploy"], output_path)

            res_deployment_pb = Deployment(state=DeploymentState())
            res_deployment_pb.CopyFrom(deployment_pb)
            state = self.describe(res_deployment_pb, repo).state
            res_deployment_pb.state.CopyFrom(state)
            return ApplyDeploymentResponse(
                status=Status.OK(), deployment=res_deployment_pb
            )
        except BentoMLException as error:
            return ApplyDeploymentResponse(status=exception_to_return_status(error))

    def delete(self, deployment_pb, repo=None):
        try:
            state = self.describe(deployment_pb, repo).state
            if state.state != DeploymentState.RUNNING:
                message = (
                    'Failed to delete, no active deployment {name}. '
                    'The current state is {state}'.format(
                        name=deployment_pb.name,
                        state=DeploymentState.State.Name(state.state),
                    )
                )
                return DeleteDeploymentResponse(status=Status.ABORTED(message))

            deployment_spec = deployment_pb.spec
            aws_config = deployment_spec.aws_lambda_operator_config

            bento_path = repo.get(
                deployment_spec.bento_name, deployment_spec.bento_version
            )
            bento_config = load_bento_service_metadata(bento_path)
            apis = generate_api_list(
                bento_config.apis, aws_config.api_name, deployment_spec.bento_name
            )

            with TemporaryServerlessConfig(
                archive_path=bento_path,
                deployment_name=deployment_pb.name,
                region=aws_config.region,
                stage=deployment_pb.namespace,
                provider_name='aws',
                functions=generate_aws_handler_functions_config(apis),
            ) as tempdir:
                response = call_serverless_command(['remove'], tempdir)
                stack_name = '{name}-{namespace}'.format(
                    name=deployment_pb.name, namespace=deployment_pb.namespace
                )
                if "Serverless: Stack removal finished..." in response:
                    status = Status.OK()
                elif "Stack '{}' does not exist".format(stack_name) in response:
                    status = Status.NOT_FOUND('Resource not found')
                else:
                    status = Status.ABORTED()

            return DeleteDeploymentResponse(status=status)
        except BentoMLException as error:
            return DeleteDeploymentResponse(status=exception_to_return_status(error))

    def describe(self, deployment_pb, repo=None):
        try:
            deployment_spec = deployment_pb.spec
            aws_config = deployment_spec.aws_lambda_operator_config
            info_json = {'endpoints': []}

            bento_path = repo.get(
                deployment_spec.bento_name, deployment_spec.bento_version
            )
            bento_config = load_bento_service_metadata(bento_path)
            apis = generate_api_list(
                bento_config.apis, aws_config.api_name, deployment_spec.bento_name
            )

            try:
                cloud_formation_stack_result = boto3.client(
                    'cloudformation'
                ).describe_stacks(
                    StackName='{name}-{ns}'.format(
                        ns=deployment_pb.namespace, name=deployment_pb.name
                    )
                )
                outputs = cloud_formation_stack_result.get('Stacks')[0]['Outputs']
            except Exception as error:
                return DescribeDeploymentResponse(
                    status=Status.INTERNAL(str(error)),
                    state=DeploymentState(
                        state=DeploymentState.ERROR, error_message=str(error)
                    ),
                )

            base_url = ''
            for output in outputs:
                if output['OutputKey'] == 'ServiceEndpoint':
                    base_url = output['OutputValue']
                    break
            if base_url:
                info_json['endpoints'] = [base_url + '/' + api.name for api in apis]
            return DescribeDeploymentResponse(
                status=Status.OK(),
                state=DeploymentState(
                    state=DeploymentState.RUNNING, info_json=json.dumps(info_json)
                ),
            )
        except BentoMLException as error:
            return DescribeDeploymentResponse(status=exception_to_return_status(error))
