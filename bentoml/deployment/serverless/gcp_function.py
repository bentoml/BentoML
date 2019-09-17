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
from bentoml.archive.loader import load_bentoml_config
from bentoml.yatai.status import Status
from bentoml.exceptions import BentoMLException
from bentoml.proto.deployment_pb2 import (
    ApplyDeploymentResponse,
    DescribeDeploymentResponse,
    DeleteDeploymentResponse,
    DeploymentState,
    Deployment,
)
from bentoml.deployment.serverless.serverless_utils import (
    call_serverless_command,
    TemporaryServerlessContent,
    TemporaryServerlessConfig,
    parse_serverless_info_response_to_json_string,
)

logger = logging.getLogger(__name__)

GOOGLE_MAIN_PY_TEMPLATE_HEADER = """\
from {class_name} import load

bento_service = load()

"""

GOOGLE_FUNCTION_TEMPLATE = """\
def {api_name}(request):
    api = bento_service.get_service_api('{api_name}')

    return api.handle_request(request)

"""


def generate_gcp_handler_functions_config(apis):
    function_list = {}
    for api in apis:
        function_list[api['name']] = {
            "handler": api['name'],
            "events": [{"http": "path"}],
        }
    return function_list


def generate_serverless_configuration_for_gcp_function(
    service_name, apis, output_path, region, stage
):
    config_path = os.path.join(output_path, "serverless.yml")
    yaml = YAML()
    with open(config_path, "r") as f:
        content = f.read()
    serverless_config = yaml.load(content)

    serverless_config["service"] = service_name
    serverless_config["provider"]["project"] = service_name

    serverless_config["provider"]["region"] = region
    logger.info("Using user defined Google region: %s", region)

    serverless_config["provider"]["stage"] = stage
    logger.info("Using user defined Google stage: %s", stage)

    serverless_config["functions"] = generate_gcp_handler_functions_config(apis)

    yaml.dump(serverless_config, Path(config_path))
    return


def generate_main_py(bento_name, apis, output_path):
    with open(os.path.join(output_path, "main.py"), "w") as f:
        f.write(GOOGLE_MAIN_PY_TEMPLATE_HEADER.format(class_name=bento_name))
        for api in apis:
            api_content = GOOGLE_FUNCTION_TEMPLATE.format(api_name=api['name'])
            f.write(api_content)
    return


class GcpFunctionDeploymentOperator(DeploymentOperatorBase):
    def apply(self, deployment_pb, repo, prev_deployment=None):
        deployment_spec = deployment_pb.spec
        gcp_config = deployment_spec.gcp_function_operator_config
        bento_path = repo.get(deployment_spec.bento_name, deployment_spec.bento_version)

        bento_config = load_bentoml_config(bento_path)
        with TemporaryServerlessContent(
            archive_path=bento_path,
            deployment_name=deployment_pb.name,
            bento_name=deployment_spec.bento_name,
            template_type='google-python',
        ) as output_path:
            generate_main_py(bento_config['name'], bento_config['apis'], output_path)
            generate_serverless_configuration_for_gcp_function(
                service_name=bento_config['name'],
                apis=bento_config['apis'],
                output_path=output_path,
                region=gcp_config.region,
                stage=deployment_pb.namespace,
            )
            call_serverless_command(["serverless", "deploy"], output_path)

        res_deployment_pb = Deployment(state=DeploymentState())
        res_deployment_pb.CopyFrom(deployment_pb)
        state = self.describe(res_deployment_pb, repo).state
        res_deployment_pb.state.CopyFrom(state)

        return ApplyDeploymentResponse(status=Status.OK(), deployment=res_deployment_pb)

    def describe(self, deployment_pb, repo=None):
        deployment_spec = deployment_pb.spec
        gcp_config = deployment_spec.gcp_function_operator_config

        bento_path = repo.get(deployment_spec.bento_name, deployment_spec.bento_version)
        bento_config = load_bentoml_config(bento_path)
        with TemporaryServerlessConfig(
            archive_path=bento_path,
            deployment_name=deployment_pb.name,
            region=gcp_config.region,
            stage=deployment_pb.namespace,
            provider_name='google',
            functions=generate_gcp_handler_functions_config(bento_config['apis']),
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

    def delete(self, deployment_pb, repo=None):
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
        gcp_config = deployment_spec.gcp_function_operator_config

        bento_path = repo.get(deployment_spec.bento_name, deployment_spec.bento_version)
        bento_config = load_bentoml_config(bento_path)
        with TemporaryServerlessConfig(
            archive_path=bento_path,
            deployment_name=deployment_pb.name,
            region=gcp_config.region,
            stage=deployment_pb.namespace,
            provider_name='google',
            functions=generate_gcp_handler_functions_config(bento_config['apis']),
        ) as tempdir:
            try:
                response = call_serverless_command(['serverless', 'remove'], tempdir)
                if "Serverless: Stack removal finished..." in response:
                    status = Status.OK()
                else:
                    status = Status.ABORTED()
            except BentoMLException as e:
                status = Status.INTERNAL(str(e))

        return DeleteDeploymentResponse(status=status)
