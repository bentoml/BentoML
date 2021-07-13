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

    def create_sagemaker_deployment(
        self,
        name,
        bento_name,
        bento_version,
        api_name,
        instance_type,
        instance_count,
        timeout,
        num_of_gunicorn_workers_per_instance=None,
        region=None,
        namespace=None,
        labels=None,
        annotations=None,
        wait=None,
        data_capture_s3_prefix=None,
        data_capture_sample_percent=None,
    ):
        """Create SageMaker deployment

        Args:
            name:
            bento_name:
            bento_version:
            api_name:
            instance_type:
            instance_count:
            timeout:
            num_of_gunicorn_workers_per_instance:
            region:
            namespace:
            labels:
            annotations:
            wait:
            data_capture_s3_prefix:
            data_capture_sample_percent:

        Returns:
            ApplyDeploymentResponse

        Raises:
            BentoMLException
        """
        deployment_pb = Deployment(
            name=name, namespace=namespace, labels=labels, annotations=annotations
        )
        deployment_pb.spec.bento_name = bento_name
        deployment_pb.spec.bento_version = bento_version
        deployment_pb.spec.operator = DeploymentSpec.AWS_SAGEMAKER
        deployment_pb.spec.sagemaker_operator_config.api_name = api_name
        deployment_pb.spec.sagemaker_operator_config.instance_count = instance_count
        deployment_pb.spec.sagemaker_operator_config.instance_type = instance_type
        deployment_pb.spec.sagemaker_operator_config.timeout = timeout

        if data_capture_s3_prefix:
            deployment_pb.spec.sagemaker_operator_config.data_capture_s3_prefix = (
                data_capture_s3_prefix
            )
        if data_capture_sample_percent:
            deployment_pb.spec.sagemaker_operator_config.data_capture_sample_percent = (
                data_capture_sample_percent
            )

        if region:
            deployment_pb.spec.sagemaker_operator_config.region = region
        if num_of_gunicorn_workers_per_instance:
            deployment_pb.spec.sagemaker_operator_config.num_of_gunicorn_workers_per_instance = (  # noqa E501
                num_of_gunicorn_workers_per_instance
            )

        return self.create(deployment_pb, wait)

    def update_sagemaker_deployment(
        self,
        deployment_name,
        namespace=None,
        api_name=None,
        instance_type=None,
        instance_count=None,
        timeout=None,
        num_of_gunicorn_workers_per_instance=None,
        bento_name=None,
        bento_version=None,
        wait=None,
        data_capture_s3_prefix=None,
        data_capture_sample_percent=None,
    ):
        """ Update current sagemaker deployment

        Args:
            namespace:
            deployment_name:
            api_name:
            instance_type:
            instance_count:
            timeout:
            num_of_gunicorn_workers_per_instance:
            bento_name:
            bento_version:
            wait:
            data_capture_s3_prefix:
            data_capture_sample_percent:

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
        if timeout:
            deployment_pb.spec.sagemaker_operator_config.timeout = timeout
        if bento_name:
            deployment_pb.spec.bento_name = bento_name
        if bento_version:
            deployment_pb.spec.bento_version = bento_version
        if data_capture_s3_prefix:
            deployment_pb.spec.sagemaker_operator_config.data_capture_s3_prefix = (
                data_capture_s3_prefix
            )
        if data_capture_sample_percent:
            deployment_pb.spec.sagemaker_operator_config.data_capture_sample_percent = (
                data_capture_sample_percent
            )

        logger.debug(
            'Updated configuration for sagemaker deployment %s', deployment_pb.name
        )

        return self.apply(deployment_pb, wait)

    def list_sagemaker_deployments(
        self,
        limit=None,
        offset=None,
        labels=None,
        namespace=None,
        is_all_namespaces=False,
        order_by=None,
        ascending_order=None,
    ):
        list_result = self.list(
            limit=limit,
            offset=offset,
            labels=labels,
            namespace=namespace,
            is_all_namespaces=is_all_namespaces,
            operator=DeploymentSpec.AWS_SAGEMAKER,
            order_by=order_by,
            ascending_order=ascending_order,
        )
        if list_result.status.status_code != status_pb2.Status.OK:
            return list_result

        sagemaker_deployments = [
            deployment
            for deployment in list_result.deployments
            if deployment.spec.operator == DeploymentSpec.AWS_SAGEMAKER
        ]
        del list_result.deployments[:]
        list_result.deployments.extend(sagemaker_deployments)
        return list_result

    def create_ec2_deployment(
        self,
        name,
        namespace,
        bento_name,
        bento_version,
        region,
        min_size,
        desired_capacity,
        max_size,
        instance_type,
        ami_id,
        wait=None,
    ):

        deployment_pb = Deployment(name=name, namespace=namespace)
        deployment_pb.spec.bento_name = bento_name
        deployment_pb.spec.bento_version = bento_version
        if region:
            deployment_pb.spec.aws_ec2_operator_config.region = region
        deployment_pb.spec.operator = DeploymentSpec.AWS_EC2
        deployment_pb.spec.aws_ec2_operator_config.autoscale_min_size = min_size
        deployment_pb.spec.aws_ec2_operator_config.autoscale_desired_capacity = (
            desired_capacity
        )
        deployment_pb.spec.aws_ec2_operator_config.autoscale_max_size = max_size
        deployment_pb.spec.aws_ec2_operator_config.instance_type = instance_type
        deployment_pb.spec.aws_ec2_operator_config.ami_id = ami_id
        return self.create(deployment_pb, wait)

    def update_ec2_deployment(
        self,
        deployment_name,
        bento_name,
        bento_version,
        namespace,
        min_size,
        desired_capacity,
        max_size,
        instance_type,
        ami_id,
        wait,
    ):
        get_deployment_result = self.get(namespace=namespace, name=deployment_name)
        if get_deployment_result.status.status_code != status_pb2.Status.OK:
            error_code = status_pb2.Status.Code.Name(
                get_deployment_result.status.status_code
            )
            error_message = get_deployment_result.status.error_message
            raise BentoMLException(
                f"Failed to retrieve current deployment {deployment_name} "
                f"in {namespace}.  {error_code}:{error_message}"
            )
        # new deloyment info with updated configs
        deployment_pb = get_deployment_result.deployment

        if bento_name:
            deployment_pb.spec.bento_name = bento_name
        if bento_version:
            deployment_pb.spec.bento_version = bento_version

        deployment_pb.spec.aws_ec2_operator_config.autoscale_min_size = min_size
        deployment_pb.spec.aws_ec2_operator_config.autoscale_desired_capacity = (
            desired_capacity
        )
        deployment_pb.spec.aws_ec2_operator_config.autoscale_max_size = max_size
        deployment_pb.spec.aws_ec2_operator_config.instance_type = instance_type
        deployment_pb.spec.aws_ec2_operator_config.ami_id = ami_id

        logger.debug("Updated configuration for Lambda deployment %s", deployment_name)

        return self.apply(deployment_pb, wait)

    def list_ec2_deployments(
        self,
        limit=None,
        offset=None,
        labels=None,
        namespace=None,
        order_by=None,
        ascending_order=None,
        is_all_namespaces=False,
    ):
        return self.list(
            limit=limit,
            offset=offset,
            labels=labels,
            namespace=namespace,
            is_all_namespaces=is_all_namespaces,
            operator=DeploymentSpec.AWS_EC2,
            order_by=order_by,
            ascending_order=ascending_order,
        )

    def create_lambda_deployment(
        self,
        name,
        bento_name,
        bento_version,
        memory_size,
        timeout,
        api_name=None,
        region=None,
        namespace=None,
        labels=None,
        annotations=None,
        wait=None,
    ):
        """Create Lambda deployment

        Args:
            name:
            bento_name:
            bento_version:
            memory_size:
            timeout:
            api_name:
            region:
            namespace:
            labels:
            annotations:
            wait:

        Returns:
            ApplyDeploymentResponse: status, deployment

        Raises:
            BentoMLException

        """
        deployment_pb = Deployment(
            name=name, namespace=namespace, labels=labels, annotations=annotations
        )
        deployment_pb.spec.bento_name = bento_name
        deployment_pb.spec.bento_version = bento_version
        deployment_pb.spec.operator = DeploymentSpec.AWS_LAMBDA
        deployment_pb.spec.aws_lambda_operator_config.memory_size = memory_size
        deployment_pb.spec.aws_lambda_operator_config.timeout = timeout
        if api_name:
            deployment_pb.spec.aws_lambda_operator_config.api_name = api_name
        if region:
            deployment_pb.spec.aws_lambda_operator_config.region = region
        return self.create(deployment_pb, wait)

    def update_lambda_deployment(
        self,
        deployment_name,
        namespace=None,
        bento_name=None,
        bento_version=None,
        memory_size=None,
        timeout=None,
        wait=None,
    ):
        get_deployment_result = self.get(namespace=namespace, name=deployment_name)
        if get_deployment_result.status.status_code != status_pb2.Status.OK:
            error_code = status_pb2.Status.Code.Name(
                get_deployment_result.status.status_code
            )
            error_message = status_pb2.status.error_message
            raise BentoMLException(
                f'Failed to retrieve current deployment {deployment_name} '
                f'in {namespace}.  {error_code}:{error_message}'
            )
        deployment_pb = get_deployment_result.deployment
        if bento_name:
            deployment_pb.spec.bento_name = bento_name
        if bento_version:
            deployment_pb.spec.bento_version = bento_version
        if memory_size:
            deployment_pb.spec.aws_lambda_operator_config.memory_size = memory_size
        if timeout:
            deployment_pb.spec.aws_lambda_operator_config.timeout = timeout
        logger.debug('Updated configuration for Lambda deployment %s', deployment_name)

        return self.apply(deployment_pb, wait)

    def list_lambda_deployments(
        self,
        limit=None,
        offset=None,
        labels=None,
        namespace=None,
        is_all_namespaces=False,
        order_by=None,
        ascending_order=None,
    ):
        return self.list(
            limit=limit,
            offset=offset,
            labels=labels,
            namespace=namespace,
            is_all_namespaces=is_all_namespaces,
            operator=DeploymentSpec.AWS_LAMBDA,
            order_by=order_by,
            ascending_order=ascending_order,
        )

    def create_azure_functions_deployment(
        self,
        name,
        bento_name,
        bento_version,
        location,
        premium_plan_sku=None,
        min_instances=None,
        max_burst=None,
        function_auth_level=None,
        namespace=None,
        labels=None,
        annotations=None,
        wait=None,
    ):
        deployment_pb = Deployment(
            name=name, namespace=namespace, labels=labels, annotations=annotations
        )

        deployment_pb.spec.bento_name = bento_name
        deployment_pb.spec.bento_version = bento_version
        deployment_pb.spec.operator = DeploymentSpec.AZURE_FUNCTIONS
        deployment_pb.spec.azure_functions_operator_config.location = location
        deployment_pb.spec.azure_functions_operator_config.premium_plan_sku = (
            premium_plan_sku
        )
        deployment_pb.spec.azure_functions_operator_config.min_instances = min_instances
        deployment_pb.spec.azure_functions_operator_config.function_auth_level = (
            function_auth_level
        )
        deployment_pb.spec.azure_functions_operator_config.max_burst = max_burst

        return self.create(deployment_pb, wait)

    def update_azure_functions_deployment(
        self,
        deployment_name,
        bento_name=None,
        bento_version=None,
        max_burst=None,
        min_instances=None,
        premium_plan_sku=None,
        namespace=None,
        wait=None,
    ):
        get_deployment_result = self.get(namespace=namespace, name=deployment_name)
        if get_deployment_result.status.status_code != status_pb2.Status.OK:
            error_code = status_pb2.Status.Code.Name(
                get_deployment_result.status.status_code
            )
            error_message = status_pb2.status.error_message
            raise BentoMLException(
                f'Failed to retrieve current deployment {deployment_name} in '
                f'{namespace}. {error_code}:{error_message}'
            )
        deployment_pb = get_deployment_result.deployment
        if bento_name:
            deployment_pb.spec.bento_name = bento_name
        if bento_version:
            deployment_pb.spec.bento_version = bento_version
        if max_burst:
            deployment_pb.spec.azure_functions_operator_config.max_burst = max_burst
        if min_instances:
            deployment_pb.spec.azure_functions_operator_config.min_instances = (
                min_instances
            )
        if premium_plan_sku:
            deployment_pb.spec.azure_functions_operator_config.premium_plan_sku = (
                premium_plan_sku
            )
        return self.apply(deployment_pb, wait)

    def list_azure_functions_deployments(
        self,
        limit=None,
        offset=None,
        labels=None,
        namespace=None,
        is_all_namespaces=False,
        order_by=None,
        ascending_order=None,
    ):
        return self.list(
            limit=limit,
            offset=offset,
            labels=labels,
            namespace=namespace,
            is_all_namespaces=is_all_namespaces,
            operator=DeploymentSpec.AZURE_FUNCTIONS,
            order_by=order_by,
            ascending_order=ascending_order,
        )
