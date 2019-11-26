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

from bentoml.deployment.utils import (
    exception_to_return_status,
    raise_if_api_names_not_found_in_bento_service_metadata,
)
from bentoml.proto.repository_pb2 import GetBentoRequest, BentoUri
from bentoml.utils import Path
from bentoml.deployment.operator import DeploymentOperatorBase
from bentoml.utils.tempdir import TempDirectory
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
    parse_serverless_info_response_to_json_string,
    init_serverless_project_dir,
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
        function_list[api.name] = {"handler": api.name, "events": [{"http": "path"}]}
    return function_list


def generate_gcp_function_serverless_config(
    deployment_name, api_names, serverless_project_dir, region, stage
):
    config_path = os.path.join(serverless_project_dir, "serverless.yml")
    if os.path.isfile(config_path):
        os.remove(config_path)
    yaml = YAML()
    serverless_config = {
        "service": deployment_name,
        "provider": {
            "region": region,
            "stage": stage,
            "name": 'google',
            "project": deployment_name,
        },
        "functions": {
            api_name: {"handler": api_name, "events": [{"http": "path"}]}
            for api_name in api_names
        },
    }

    yaml.dump(serverless_config, Path(config_path))


def generate_gcp_function_main_py(bento_name, api_names, output_path):
    with open(os.path.join(output_path, "main.py"), "w") as f:
        f.write(GOOGLE_MAIN_PY_TEMPLATE_HEADER.format(class_name=bento_name))
        for api_name in api_names:
            api_content = GOOGLE_FUNCTION_TEMPLATE.format(api_name=api_name)
            f.write(api_content)


class GcpFunctionDeploymentOperator(DeploymentOperatorBase):
    def apply(self, deployment_pb, yatai_service, prev_deployment=None):
        try:
            deployment_spec = deployment_pb.spec
            gcp_config = deployment_spec.gcp_function_operator_config
            bento_pb = yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            if bento_pb.bento.uri.type != BentoUri.LOCAL:
                raise BentoMLException(
                    'BentoML currently only support local repository'
                )
            else:
                bento_path = bento_pb.bento.uri.uri
            bento_service_metadata = bento_pb.bento.bento_service_metadata

            api_names = (
                [gcp_config.api_name]
                if gcp_config.api_name
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
                    'google-python',
                )
                generate_gcp_function_main_py(
                    deployment_spec.bento_name, api_names, serverless_project_dir
                )
                generate_gcp_function_serverless_config(
                    deployment_pb.name,
                    api_names,
                    serverless_project_dir,
                    gcp_config.region,
                    # BentoML namespace is mapping to serverless stage.
                    stage=deployment_pb.namespace,
                )
                call_serverless_command(["deploy"], serverless_project_dir)

            res_deployment_pb = Deployment(state=DeploymentState())
            res_deployment_pb.CopyFrom(deployment_pb)
            state = self.describe(res_deployment_pb, yatai_service).state
            res_deployment_pb.state.CopyFrom(state)

            return ApplyDeploymentResponse(
                status=Status.OK(), deployment=res_deployment_pb
            )
        except BentoMLException as error:
            return ApplyDeploymentResponse(status=exception_to_return_status(error))

    def describe(self, deployment_pb, yatai_service=None):
        try:
            deployment_spec = deployment_pb.spec
            gcp_config = deployment_spec.gcp_function_operator_config

            bento_pb = yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            bento_service_metadata = bento_pb.bento.bento_service_metadata
            api_names = (
                [gcp_config.api_name]
                if gcp_config.api_name
                else [api.name for api in bento_service_metadata.apis]
            )
            with TempDirectory() as serverless_project_dir:
                generate_gcp_function_serverless_config(
                    deployment_pb.name,
                    api_names,
                    serverless_project_dir,
                    gcp_config.region,
                    # BentoML namespace is mapping to serverless stage.
                    stage=deployment_pb.namespace,
                )
                try:
                    response = call_serverless_command(["info"], serverless_project_dir)
                    info_json = parse_serverless_info_response_to_json_string(response)
                    state = DeploymentState(
                        state=DeploymentState.RUNNING, info_json=info_json
                    )
                    state.timestamp.GetCurrentTime()
                except BentoMLException as e:
                    state = DeploymentState(
                        state=DeploymentState.ERROR, error_message=str(e)
                    )
                    state.timestamp.GetCurrentTime()

            return DescribeDeploymentResponse(status=Status.OK(), state=state)
        except BentoMLException as error:
            return DescribeDeploymentResponse(status=exception_to_return_status(error))

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
            gcp_config = deployment_spec.gcp_function_operator_config

            bento_pb = yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            bento_service_metadata = bento_pb.bento.bento_service_metadata
            api_names = (
                [gcp_config.api_name]
                if gcp_config.api_name
                else [api.name for api in bento_service_metadata.apis]
            )
            with TempDirectory() as serverless_project_dir:
                generate_gcp_function_serverless_config(
                    deployment_pb.name,
                    api_names,
                    serverless_project_dir,
                    gcp_config.region,
                    # BentoML namespace is mapping to serverless stage.
                    stage=deployment_pb.namespace,
                )
                try:
                    response = call_serverless_command(
                        ['remove'], serverless_project_dir
                    )
                    if "Serverless: Stack removal finished..." in response:
                        status = Status.OK()
                    else:
                        status = Status.ABORTED()
                except BentoMLException as e:
                    status = Status.INTERNAL(str(e))

            return DeleteDeploymentResponse(status=status)
        except BentoMLException as error:
            return DeleteDeploymentResponse(status=exception_to_return_status(error))
