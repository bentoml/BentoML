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

# List of APIs for accessing remote or local yatai service via Python


import logging
import time

from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.yatai.client.label_utils import generate_gprc_labels_selector
from bentoml.yatai.deployment import ALL_NAMESPACE_TAG
from bentoml.yatai.proto.deployment_pb2 import (
    ApplyDeploymentRequest,
    DescribeDeploymentRequest,
    GetDeploymentRequest,
    DeploymentSpec,
    DeleteDeploymentRequest,
    ListDeploymentsRequest,
    Deployment,
    DeploymentState,
)
from bentoml.exceptions import BentoMLException, YataiDeploymentException
from bentoml.yatai.proto import status_pb2
from bentoml.yatai.deployment_utils import (
    deployment_yaml_string_to_pb,
    deployment_dict_to_pb,
)

logger = logging.getLogger(__name__)

WAIT_TIMEOUT_LIMIT = 600
WAIT_TIME = 5


class DeploymentAPIClient:
    def __init__(self, yatai_service):
        self.yatai_service = yatai_service

    def list(
        self,
        limit=None,
        offset=None,
        labels=None,
        namespace=None,
        is_all_namespaces=False,
        operator=None,
        order_by=None,
        ascending_order=None,
    ):
        if is_all_namespaces:
            if namespace is not None:
                logger.warning(
                    'Ignoring `namespace=%s` due to the --all-namespace flag presented',
                    namespace,
                )
            namespace = ALL_NAMESPACE_TAG
        if isinstance(operator, str):
            if operator == 'sagemaker':
                operator = DeploymentSpec.AWS_SAGEMAKER
            elif operator == 'lambda':
                operator = DeploymentSpec.AWS_LAMBDA
            elif operator == DeploymentSpec.AZURE_FUNCTIONS:
                operator = 'azure-functions'
            elif operator == "ec2":
                operator = DeploymentSpec.AWS_EC2
            else:
                raise BentoMLException(f'Unrecognized operator {operator}')

        list_deployment_request = ListDeploymentsRequest(
            limit=limit,
            offset=offset,
            namespace=namespace,
            operator=operator,
            order_by=order_by,
            ascending_order=ascending_order,
        )
        if labels is not None:
            generate_gprc_labels_selector(
                list_deployment_request.label_selectors, labels
            )
        return self.yatai_service.ListDeployments(list_deployment_request)

    def get(self, namespace, name):
        return self.yatai_service.GetDeployment(
            GetDeploymentRequest(deployment_name=name, namespace=namespace)
        )

    def describe(self, namespace, name):
        return self.yatai_service.DescribeDeployment(
            DescribeDeploymentRequest(deployment_name=name, namespace=namespace)
        )

    def delete(self, deployment_name, namespace, force_delete=False):
        return self.yatai_service.DeleteDeployment(
            DeleteDeploymentRequest(
                deployment_name=deployment_name,
                namespace=namespace,
                force_delete=force_delete,
            )
        )

    def create(self, deployment_info, wait):
        from bentoml.yatai.validator import validate_deployment_pb

        if isinstance(deployment_info, dict):
            deployment_pb = deployment_dict_to_pb(deployment_info)
        elif isinstance(deployment_info, str):
            deployment_pb = deployment_yaml_string_to_pb(deployment_info)
        elif isinstance(deployment_info, Deployment):
            deployment_pb = deployment_info
        else:
            raise YataiDeploymentException(
                'Unexpected argument type, expect deployment info to be str in yaml '
                'format or a dict or a deployment protobuf obj, instead got: {}'.format(
                    str(type(deployment_info))
                )
            )

        validation_errors = validate_deployment_pb(deployment_pb)
        if validation_errors:
            raise YataiDeploymentException(
                f'Failed to validate deployment {deployment_pb.name}: '
                f'{validation_errors}'
            )
        # Make sure there is no active deployment with the same deployment name
        get_deployment_pb = self.yatai_service.GetDeployment(
            GetDeploymentRequest(
                deployment_name=deployment_pb.name, namespace=deployment_pb.namespace
            )
        )
        if get_deployment_pb.status.status_code != status_pb2.Status.NOT_FOUND:
            raise BentoMLException(
                f'Deployment "{deployment_pb.name}" already existed, use Update or '
                f'Apply for updating existing deployment, delete the deployment, '
                f'or use a different deployment name'
            )
        apply_result = self.yatai_service.ApplyDeployment(
            ApplyDeploymentRequest(deployment=deployment_pb)
        )
        if apply_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                apply_result.status
            )
            raise YataiDeploymentException(f'{error_code}:{error_message}')
        if wait:
            self._wait_deployment_action_complete(
                deployment_pb.name, deployment_pb.namespace
            )
        return self.get(namespace=deployment_pb.namespace, name=deployment_pb.name)

    def apply(self, deployment_info, wait):
        from bentoml.yatai.validator import validate_deployment_pb

        if isinstance(deployment_info, dict):
            deployment_pb = deployment_dict_to_pb(deployment_info)
        elif isinstance(deployment_info, str):
            deployment_pb = deployment_yaml_string_to_pb(deployment_info)
        elif isinstance(deployment_info, Deployment):
            deployment_pb = deployment_info
        else:
            raise YataiDeploymentException(
                'Unexpected argument type, expect deployment info to be str in yaml '
                'format or a dict or a deployment protobuf obj, instead got: {}'.format(
                    str(type(deployment_info))
                )
            )

        validation_errors = validate_deployment_pb(deployment_pb)
        if validation_errors:
            raise YataiDeploymentException(
                f'Failed to validate deployment {deployment_pb.name}: '
                f'{validation_errors}'
            )

        apply_result = self.yatai_service.ApplyDeployment(
            ApplyDeploymentRequest(deployment=deployment_pb)
        )
        if apply_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                apply_result.status
            )
            raise YataiDeploymentException(f'{error_code}:{error_message}')
        if wait:
            self._wait_deployment_action_complete(
                deployment_pb.name, deployment_pb.namespace
            )
        return self.get(namespace=deployment_pb.namespace, name=deployment_pb.name)

    def _wait_deployment_action_complete(self, name, namespace):
        start_time = time.time()
        while (time.time() - start_time) < WAIT_TIMEOUT_LIMIT:
            result = self.describe(namespace=namespace, name=name)
            if (
                result.status.status_code == status_pb2.Status.OK
                and result.state.state is DeploymentState.PENDING
            ):
                time.sleep(WAIT_TIME)
                continue
            else:
                break
        return result
