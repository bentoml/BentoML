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

from bentoml.bundler import loader
from bentoml.exceptions import BentoMLException
from bentoml.utils import Path
from bentoml.deployment.operator import DeploymentOperatorBase
from bentoml.utils.tempdir import TempDirectory
from bentoml.yatai.status import Status
from bentoml.proto.deployment_pb2 import (
    Deployment,
    ApplyDeploymentResponse,
    DescribeDeploymentResponse,
    DeleteDeploymentResponse,
    DeploymentState,
)
from bentoml.proto.repository_pb2 import GetBentoRequest, BentoUri
from bentoml.deployment.utils import (
    ensure_docker_available_or_raise,
    exception_to_return_status,
    raise_if_api_names_not_found_in_bento_service_metadata,
)
from bentoml.deployment.serverless.serverless_utils import (
    call_serverless_command,
    install_serverless_plugin,
    init_serverless_project_dir,
)

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


def generate_aws_lambda_serverless_config(
    bento_python_version,
    deployment_name,
    api_names,
    serverless_project_dir,
    region,
    stage,
):
    config_path = os.path.join(serverless_project_dir, "serverless.yml")
    if os.path.isfile(config_path):
        os.remove(config_path)
    yaml = YAML()

    runtime = 'python3.7'
    if version.parse(bento_python_version) < version.parse('3.0.0'):
        runtime = 'python2.7'

    serverless_config = {
        "service": deployment_name,
        "provider": {
            "region": region,
            "stage": stage,
            "name": 'aws',
            'runtime': runtime,
            "apiGateway": {"binaryMediaTypes": ['image/*']},
        },
        "functions": {
            api_name: {
                "handler": "handler." + api_name,
                "events": [{"http": {"path": "/" + api_name, "method": "post"}}],
            }
            for api_name in api_names
        },
        "custom": {
            "pythonRequirements": {
                "useDownloadCache": True,
                "useStaticCache": True,
                "dockerizePip": True,
                "slim": True,
                "strip": True,
                "zip": True,
                # We are passing the bundled_pip_dependencies directory for python
                # requirement package, so it can installs the bundled tar gz file.
                "dockerRunCmdExtraArgs": [
                    '-v',
                    '{}/bundled_pip_dependencies:'
                    '/var/task/bundled_pip_dependencies:z'.format(
                        serverless_project_dir
                    ),
                ],
            }
        },
    }

    yaml.dump(serverless_config, Path(config_path))


def generate_aws_lambda_handler_py(bento_name, api_names, output_path):
    with open(os.path.join(output_path, "handler.py"), "w") as f:
        f.write(AWS_HANDLER_PY_TEMPLATE_HEADER.format(class_name=bento_name))
        for api_name in api_names:
            api_content = AWS_FUNCTION_TEMPLATE.format(api_name=api_name)
            f.write(api_content)


class AwsLambdaDeploymentOperator(DeploymentOperatorBase):
    def apply(self, deployment_pb, yatai_service, prev_deployment=None):
        try:
            ensure_docker_available_or_raise()
            deployment_spec = deployment_pb.spec

            bento_pb = yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            if bento_pb.bento.uri.type not in (BentoUri.LOCAL, BentoUri.S3):
                raise BentoMLException(
                    'BentoML currently not support {} repository'.format(
                        bento_pb.bento.uri.type
                    )
                )

            return self._apply(
                deployment_pb, bento_pb, yatai_service, bento_pb.bento.uri.uri
            )

        except BentoMLException as error:
            return ApplyDeploymentResponse(status=exception_to_return_status(error))

    def _apply(self, deployment_pb, bento_pb, yatai_service, bento_path):
        if loader._is_remote_path(bento_path):
            with loader._resolve_remote_bundle_path(bento_path) as local_path:
                return self._apply(deployment_pb, bento_pb, yatai_service, local_path)

        deployment_spec = deployment_pb.spec
        aws_config = deployment_spec.aws_lambda_operator_config

        bento_service_metadata = bento_pb.bento.bento_service_metadata

        template = 'aws-python3'
        if version.parse(bento_service_metadata.env.python_version) < version.parse(
            '3.0.0'
        ):
            template = 'aws-python'

        api_names = (
            [aws_config.api_name]
            if aws_config.api_name
            else [api.name for api in bento_service_metadata.apis]
        )
        raise_if_api_names_not_found_in_bento_service_metadata(
            bento_service_metadata, api_names
        )

        with TempDirectory() as serverless_project_dir:
            init_serverless_project_dir(
                serverless_project_dir,
                bento_path,
                deployment_pb.name,
                deployment_spec.bento_name,
                template,
            )
            generate_aws_lambda_handler_py(
                deployment_spec.bento_name, api_names, serverless_project_dir
            )
            generate_aws_lambda_serverless_config(
                bento_service_metadata.env.python_version,
                deployment_pb.name,
                api_names,
                serverless_project_dir,
                aws_config.region,
                # BentoML deployment namespace is mapping to serverless `stage`
                # concept
                stage=deployment_pb.namespace,
            )
            logger.info(
                'Installing additional packages: serverless-python-requirements'
            )
            install_serverless_plugin(
                "serverless-python-requirements", serverless_project_dir
            )
            logger.info('Deploying to AWS Lambda')
            call_serverless_command(["deploy"], serverless_project_dir)

        res_deployment_pb = Deployment(state=DeploymentState())
        res_deployment_pb.CopyFrom(deployment_pb)
        state = self.describe(res_deployment_pb, yatai_service).state
        res_deployment_pb.state.CopyFrom(state)
        return ApplyDeploymentResponse(status=Status.OK(), deployment=res_deployment_pb)

    def delete(self, deployment_pb, yatai_service=None):
        try:
            state = self.describe(deployment_pb, yatai_service).state
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

            bento_pb = yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            bento_service_metadata = bento_pb.bento.bento_service_metadata
            # We are not validating api_name, because for delete, you don't
            # need them.
            api_names = (
                [aws_config.api_name]
                if aws_config.api_name
                else [api.name for api in bento_service_metadata.apis]
            )

            with TempDirectory() as serverless_project_dir:
                generate_aws_lambda_serverless_config(
                    bento_service_metadata.env.python_version,
                    deployment_pb.name,
                    api_names,
                    serverless_project_dir,
                    aws_config.region,
                    # BentoML deployment namespace is mapping to serverless `stage`
                    # concept
                    stage=deployment_pb.namespace,
                )
                response = call_serverless_command(['remove'], serverless_project_dir)
                stack_name = '{name}-{namespace}'.format(
                    name=deployment_pb.name, namespace=deployment_pb.namespace
                )
                if "Serverless: Stack removal finished..." in response:
                    status = Status.OK()
                elif "Stack '{}' does not exist".format(stack_name) in response:
                    status = Status.NOT_FOUND(
                        'Deployment {} not found'.format(stack_name)
                    )
                else:
                    status = Status.ABORTED()

            return DeleteDeploymentResponse(status=status)
        except BentoMLException as error:
            return DeleteDeploymentResponse(status=exception_to_return_status(error))

    def describe(self, deployment_pb, yatai_service=None):
        try:
            deployment_spec = deployment_pb.spec
            aws_config = deployment_spec.aws_lambda_operator_config
            info_json = {'endpoints': []}

            bento_pb = yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            bento_service_metadata = bento_pb.bento.bento_service_metadata
            api_names = (
                [aws_config.api_name]
                if aws_config.api_name
                else [api.name for api in bento_service_metadata.apis]
            )

            try:
                cf_client = boto3.client('cloudformation', aws_config.region)
                cloud_formation_stack_result = cf_client.describe_stacks(
                    StackName='{name}-{ns}'.format(
                        ns=deployment_pb.namespace, name=deployment_pb.name
                    )
                )
                outputs = cloud_formation_stack_result.get('Stacks')[0]['Outputs']
            except Exception as error:
                state = DeploymentState(
                    state=DeploymentState.ERROR, error_message=str(error)
                )
                state.timestamp.GetCurrentTime()
                return DescribeDeploymentResponse(
                    status=Status.INTERNAL(str(error)), state=state
                )

            base_url = ''
            for output in outputs:
                if output['OutputKey'] == 'ServiceEndpoint':
                    base_url = output['OutputValue']
                    break
            if base_url:
                info_json['endpoints'] = [
                    base_url + '/' + api_name for api_name in api_names
                ]
            state = DeploymentState(
                state=DeploymentState.RUNNING, info_json=json.dumps(info_json)
            )
            state.timestamp.GetCurrentTime()
            return DescribeDeploymentResponse(status=Status.OK(), state=state)
        except BentoMLException as error:
            return DescribeDeploymentResponse(status=exception_to_return_status(error))
