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

from bentoml import config
from bentoml.deployment.store import ALL_NAMESPACE_TAG
from bentoml.proto.deployment_pb2 import (
    ApplyDeploymentRequest,
    DescribeDeploymentRequest,
    GetDeploymentRequest,
    DeploymentSpec,
    DeleteDeploymentRequest,
    ListDeploymentsRequest,
    ApplyDeploymentResponse,
    Deployment,
)
from bentoml.exceptions import BentoMLException, YataiDeploymentException
from bentoml.proto import status_pb2
from bentoml.utils.validator import validate_deployment_pb_schema
from bentoml.yatai.deployment_utils import (
    deployment_yaml_string_to_pb,
    deployment_dict_to_pb,
)
from bentoml.yatai.status import Status

logger = logging.getLogger(__name__)


class DeploymentAPIClient:
    def __init__(self, yatai_service):
        self.yatai_service = yatai_service

    def list(
        self,
        limit=None,
        filters=None,
        labels=None,
        namespace=None,
        is_all_namespaces=False,
    ):
        if is_all_namespaces:
            if namespace is not None:
                logger.warning(
                    'Ignoring `namespace=%s` due to the --all-namespace flag presented',
                    namespace,
                )
            namespace = ALL_NAMESPACE_TAG

        return self.yatai_service.ListDeployments(
            ListDeploymentsRequest(
                limit=limit, filter=filters, labels=labels, namespace=namespace
            )
        )

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

    def apply_deployment(self, deployment_info):
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

        validation_errors = validate_deployment_pb_schema(deployment_pb)
        if validation_errors:
            return ApplyDeploymentResponse(
                status=Status.INVALID_ARGUMENT(
                    'Failed to validate deployment: {errors}'.format(
                        errors=validation_errors
                    )
                )
            )

        return self.yatai_service.ApplyDeployment(
            ApplyDeploymentRequest(deployment=deployment_pb)
        )

    def create(
        self,
        deployment_name,
        namespace,
        bento_name,
        bento_version,
        platform,
        operator_spec,
        labels=None,
        annotations=None,
    ):
        # Make sure there is no active deployment with the same deployment name
        get_deployment_pb = self.yatai_service.GetDeployment(
            GetDeploymentRequest(deployment_name=deployment_name, namespace=namespace)
        )
        if get_deployment_pb.status.status_code == status_pb2.Status.OK:
            raise YataiDeploymentException(
                'Deployment "{name}" already existed, use Update or Apply for updating '
                'existing deployment, delete the deployment, or use a different '
                'deployment name'.format(name=deployment_name)
            )
        if get_deployment_pb.status.status_code != status_pb2.Status.NOT_FOUND:
            raise YataiDeploymentException(
                'Failed accesing YataiService deployment store. {error_code}:'
                '{error_message}'.format(
                    error_code=Status.Name(get_deployment_pb.status.status_code),
                    error_message=get_deployment_pb.status.error_message,
                )
            )

        deployment_dict = {
            "name": deployment_name,
            "namespace": namespace or config().get('deployment', 'default_namespace'),
            "labels": labels,
            "annotations": annotations,
            "spec": {
                "bento_name": bento_name,
                "bento_version": bento_version,
                "operator": platform,
            },
        }

        operator = platform.replace('-', '_').upper()
        try:
            operator_value = DeploymentSpec.DeploymentOperator.Value(operator)
        except ValueError:
            return ApplyDeploymentResponse(
                status=Status.INVALID_ARGUMENT('Invalid platform "{}"'.format(platform))
            )
        if operator_value == DeploymentSpec.AWS_SAGEMAKER:
            deployment_dict['spec']['sagemaker_operator_config'] = {
                'region': operator_spec.get('region')
                or config().get('aws', 'default_region'),
                'instance_count': operator_spec.get('instance_count'),
                'instance_type': operator_spec.get('instance_type'),
                'api_name': operator_spec.get('api_name', ''),
            }
            if operator_spec.get('num_of_gunicorn_workers_per_instance'):
                deployment_dict['spec']['sagemaker_operator_config'][
                    'num_of_gunicorn_workers_per_instance'
                ] = operator_spec.get('num_of_gunicorn_workers_per_instance')
        elif operator_value == DeploymentSpec.AWS_LAMBDA:
            deployment_dict['spec']['aws_lambda_operator_config'] = {
                'region': operator_spec.get('region')
                or config().get('aws', 'default_region')
            }
            for field in ['api_name', 'memory_size', 'timeout']:
                if operator_spec.get(field):
                    deployment_dict['spec']['aws_lambda_operator_config'][
                        field
                    ] = operator_spec[field]
        elif operator_value == DeploymentSpec.KUBERNETES:
            deployment_dict['spec']['kubernetes_operator_config'] = {
                'kube_namespace': operator_spec.get('kube_namespace', ''),
                'replicas': operator_spec.get('replicas', 0),
                'service_name': operator_spec.get('service_name', ''),
                'service_type': operator_spec.get('service_type', ''),
            }
        else:
            raise YataiDeploymentException(
                'Platform "{}" is not supported in the current version of '
                'BentoML'.format(platform)
            )

        apply_response = self.apply(deployment_dict)

        if apply_response.status.status_code == status_pb2.Status.OK:
            describe_response = self.describe(deployment_name, namespace)
            if describe_response.status.status_code == status_pb2.Status.OK:
                deployment_state = describe_response.state
                apply_response.deployment.state.CopyFrom(deployment_state)
                return apply_response

        return apply_response

    def update_sagemaker_deployment(
        self,
        namespace,
        deployment_name,
        api_name=None,
        instance_type=None,
        instance_count=None,
        num_of_gunicorn_workers_per_instance=None,
        bento_name=None,
        bento_version=None,
    ):
        """ Update current sagemaker deployment

        Args:
            namespace:
            deployment_name:
            api_name:
            instance_type:
            instance_count:
            num_of_gunicorn_workers_per_instance:
            bento_name:
            bento_version:
            yatai_service:

        Returns:
            Protobuf message

        Raises:
             BentoMLException
        """

        get_deployment_result = self.get(namespace, deployment_name)
        if get_deployment_result.status.status_code != status_pb2.Status.OK:
            get_deployment_status = get_deployment_result.status
            raise BentoMLException(
                f'Failed to retrieve current deployment {deployment_name} in '
                f'{namespace}. '
                f'{status_pb2.Status.Code.Name(get_deployment_status.status_code)}'
                f':{get_deployment_status.error_message}'
            )

        deployment_pb = get_deployment_result.deployment
        if api_name:
            deployment_pb.spec.sagemaker_operator_config.api_name = api_name
        if instance_type:
            deployment_pb.spec.sagemaker_operator_config.instance_type = instance_type
        if instance_count:
            deployment_pb.spec.sagemaker_operator_config.instance_count = instance_count
        if num_of_gunicorn_workers_per_instance:
            deployment_pb.spec.sagemaker_operator_config.num_of_gunicorn_workers_per_instance = (  # noqa E501
                num_of_gunicorn_workers_per_instance
            )
        if bento_name:
            deployment_pb.spec.bento_name = bento_name
        if bento_version:
            deployment_pb.spec.bento_version = bento_version

        logger.debug(
            'Updated configuration for sagemaker deployment %s', deployment_pb.name
        )

        apply_response = self.apply(deployment_pb)

        if apply_response.status.status_code == status_pb2.Status.OK:
            describe_response = self.describe(
                deployment_pb.name, deployment_pb.namespace
            )
            if describe_response.status.status_code == status_pb2.Status.OK:
                apply_response.deployment.state.CopyFrom(describe_response.state)
                return apply_response
        return apply_response
