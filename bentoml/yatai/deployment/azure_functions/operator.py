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
import json
import os
import re
import shutil
import subprocess
import logging
import sys

import docker

from bentoml.utils.tempdir import TempDirectory
from bentoml.saved_bundle import loader
from bentoml.yatai.deployment.azure_functions.constants import (
    MAX_RESOURCE_GROUP_NAME_LENGTH,
    MAX_STORAGE_ACCOUNT_NAME_LENGTH,
    MAX_FUNCTION_NAME_LENGTH,
    MAX_CONTAINER_REGISTRY_NAME_LENGTH,
    DEFAULT_MIN_INSTANCE_COUNT,
    DEFAULT_MAX_BURST,
    DEFAULT_PREMIUM_PLAN_SKU,
    DEFAULT_FUNCTION_AUTH_LEVEL,
)
from bentoml.yatai.deployment.azure_functions.templates import AZURE_API_FUNCTION_JSON
from bentoml.yatai.deployment.operator import DeploymentOperatorBase
from bentoml.yatai.deployment.docker_utils import ensure_docker_available_or_raise
from bentoml.exceptions import (
    BentoMLException,
    MissingDependencyException,
    AzureServiceError,
    YataiDeploymentException,
)
from bentoml.yatai.proto.deployment_pb2 import (
    ApplyDeploymentResponse,
    DeploymentState,
    DescribeDeploymentResponse,
    DeleteDeploymentResponse,
)
from bentoml.yatai.proto.repository_pb2 import GetBentoRequest
from bentoml.yatai.status import Status
from bentoml.configuration import LAST_PYPI_RELEASE_VERSION

logger = logging.getLogger(__name__)


def _assert_az_cli_logged_in():
    account_info = _call_az_cli(
        command=['az', 'account', 'show'], message='show Azure account'
    )
    if not account_info['user'] or not account_info['homeTenantId']:
        raise YataiDeploymentException(
            'A signed in Azure CLI is required for Azure Function deployment'
        )


def _assert_azure_cli_available():
    try:
        _call_az_cli(command=['az', 'account', 'show'], message='show Azure account')
    except FileNotFoundError:
        raise MissingDependencyException(
            'azure cli is required for this deployment. Please visit '
            'https://docs.microsoft.com/en-us/cli/azure/install-azure-cli '
            'for instructions'
        )


def _init_azure_functions_project(
    azure_functions_project_dir, bento_path, azure_functions_config
):
    try:
        local_bentoml_path = os.path.dirname(__file__)
        shutil.copytree(bento_path, azure_functions_project_dir)
        shutil.copy(
            os.path.join(local_bentoml_path, 'host.json'),
            os.path.join(azure_functions_project_dir, 'host.json'),
        )
        shutil.copy(
            os.path.join(local_bentoml_path, 'local.settings.json'),
            os.path.join(azure_functions_project_dir, 'local.settings.json'),
        )
        shutil.copy(
            os.path.join(local_bentoml_path, 'Dockerfile'),
            os.path.join(azure_functions_project_dir, 'Dockerfile-azure'),
        )

        app_path = os.path.join(azure_functions_project_dir, 'app')
        os.mkdir(app_path)
        shutil.copy(
            os.path.join(local_bentoml_path, 'app_init.py'),
            os.path.join(app_path, '__init__.py'),
        )
        with open(os.path.join(app_path, 'function.json'), 'w') as f:
            f.write(
                AZURE_API_FUNCTION_JSON.format(
                    function_auth_level=azure_functions_config.function_auth_level
                    or DEFAULT_FUNCTION_AUTH_LEVEL
                )
            )
    except Exception as e:
        raise BentoMLException(f'Failed to initialize azure function project. {str(e)}')


def _generate_azure_resource_names(namespace, deployment_name):
    # Generate resource names base on
    # https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/resource-name-rules

    # 1-90, alphanumeric(A-Za-z0-9) underscores, parentheses, hyphens, periods
    # scope subscription
    resource_group_name = f'{namespace}-{deployment_name}'
    if len(resource_group_name) > MAX_RESOURCE_GROUP_NAME_LENGTH:
        resource_group_name = f'{namespace[:29]}-{deployment_name[:60]}'

    # 3-24 a-z0-9, scope: global
    storage_account_name = f'{namespace}{deployment_name}'.lower()
    if len(storage_account_name) > MAX_STORAGE_ACCOUNT_NAME_LENGTH:
        storage_account_name = f'{namespace[:5]}{deployment_name[0:19]}'
    # Replace invalid chars in storage account name to '0'
    storage_account_name = re.sub(re.compile("[^a-z0-9]"), '0', storage_account_name)

    # Azure has no documentation on the requirements for function plan name.
    function_plan_name = deployment_name

    # same as Microsoft.Web/sites
    # 2-60, alphanumeric and hyphens. scope global
    function_name = f'{namespace}-{deployment_name}'
    if len(deployment_name) > MAX_FUNCTION_NAME_LENGTH:
        function_name = f'{namespace[:19]}-{deployment_name[:40]}'
    function_name = re.sub(re.compile("[^a-zA-Z0-9-]"), '-', function_name)

    # 5-50, alphanumeric scope global
    container_registry_name = f'{namespace}{deployment_name}'
    if len(container_registry_name) > MAX_CONTAINER_REGISTRY_NAME_LENGTH:
        container_registry_name = f'{namespace[:10]}{deployment_name[:40]}'
    container_registry_name = re.sub(
        re.compile("[^a-zA-Z0-9]"), '0', container_registry_name
    )

    return (
        resource_group_name,
        storage_account_name,
        function_plan_name,
        function_name,
        container_registry_name,
    )


def _call_az_cli(command, message, parse_json=True):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sys_default_encoding = sys.getfilesystemencoding()
    stdout, stderr = proc.communicate()
    if proc.returncode == 0:
        result = stdout.decode(sys_default_encoding)
        if result.endswith('\x1b[0m'):
            # remove console color code: \x1b[0m
            # https://github.com/Azure/azure-cli/issues/9903
            result = result.replace('\x1b[0m', '')
        logger.debug(f'AZ command "{" ".join(command)}" Result: {result}')
        if parse_json:
            return json.loads(result)
        else:
            return result
    else:
        error_message = stderr.decode(sys_default_encoding)
        if not error_message:
            error_message = stdout.decode(sys_default_encoding)
        logger.error(f'AZ command: "{" ".join(command)}" failed: {error_message}')
        raise AzureServiceError(f'Failed to {message}. {str(error_message)}')


def _login_acr_registry(acr_name, resource_group_name):
    result = _call_az_cli(
        command=[
            'az',
            'acr',
            'login',
            '--name',
            acr_name,
            '--resource-group',
            resource_group_name,
        ],
        message='log into Azure container registry',
        parse_json=False,
    ).replace('\n', '')
    if result == 'Login Succeed':
        raise AzureServiceError(
            f'Failed to log into Azure container registry. {result}'
        )


def _build_and_push_docker_image_to_azure_container_registry(
    azure_functions_project_dir,
    container_registry_name,
    resource_group_name,
    bento_name,
    bento_version,
    bento_python_version,
):
    _login_acr_registry(container_registry_name, resource_group_name)
    docker_client = docker.from_env()
    major, minor, _ = bento_python_version.split('.')
    try:
        docker_client.ping()
    except docker.errors.APIError as err:
        raise YataiDeploymentException(
            f'Failed to get response from docker server: {str(err)}'
        )
    tag = f'{container_registry_name}.azurecr.io/{bento_name}:{bento_version}'.lower()
    logger.debug(f'Building docker image {tag}')
    try:
        docker_client.images.build(
            path=azure_functions_project_dir,
            dockerfile=os.path.join(azure_functions_project_dir, 'Dockerfile-azure'),
            tag=tag,
            buildargs={
                'BENTOML_VERSION': LAST_PYPI_RELEASE_VERSION,
                'PYTHON_VERSION': major + minor,
            },
        )
        logger.debug('Finished building docker image')
    except docker.errors.BuildError as e:
        raise YataiDeploymentException(
            f'Failed to build docker image. BuildError: {str(e)}'
        )
    except docker.errors.APIError as e:
        raise YataiDeploymentException(
            f'Failed to build docker image. APIError: {str(e)}'
        )
    logger.debug(f'Pushing docker image {tag}')
    try:
        docker_client.images.push(tag)
        logger.debug('Finished pushing docker image')
    except docker.errors.APIError as e:
        raise YataiDeploymentException(
            f'Failed to push docker image. APIError: {str(e)}'
        )

    return tag


def _get_docker_login_info(resource_group_name, container_registry_name):
    _call_az_cli(
        command=[
            'az',
            'acr',
            'update',
            '--name',
            container_registry_name,
            '--admin-enabled',
            'true',
        ],
        message='enable admin for Azure container registry',
    )
    docker_login_info = _call_az_cli(
        command=[
            'az',
            'acr',
            'credential',
            'show',
            '--name',
            container_registry_name,
            '--resource-group',
            resource_group_name,
        ],
        message='show Azure container registry credential info',
    )

    return docker_login_info['username'], docker_login_info['passwords'][0]['value']


def _set_cors_settings(function_name, resource_group_name):
    # To allow all, use `*` and  remove all other origins in the list.
    cors_list_result = _call_az_cli(
        command=[
            'az',
            'functionapp',
            'cors',
            'show',
            '--name',
            function_name,
            '--resource-group',
            resource_group_name,
        ],
        message='show Azure functionapp cors settings',
    )
    for origin_url in cors_list_result['allowedOrigins']:
        _call_az_cli(
            command=[
                'az',
                'functionapp',
                'cors',
                'remove',
                '--name',
                function_name,
                '--resource-group',
                resource_group_name,
                '--allowed-origins',
                origin_url,
            ],
            message=f'remove allowed origin "{origin_url}"from Azure functionapp',
        )

    _call_az_cli(
        command=[
            'az',
            'functionapp',
            'cors',
            'add',
            '--name',
            function_name,
            '--resource-group',
            resource_group_name,
            '--allowed-origins',
            '*',
        ],
        message='update Azure functionapp cors setting',
    )


def _deploy_azure_functions(
    namespace, deployment_name, deployment_spec, bento_pb, bento_path,
):
    azure_functions_config = deployment_spec.azure_functions_operator_config
    (
        resource_group_name,
        storage_account_name,
        function_plan_name,
        function_name,
        container_registry_name,
    ) = _generate_azure_resource_names(namespace, deployment_name)
    with TempDirectory() as temp_dir:
        azure_functions_project_dir = os.path.join(temp_dir, deployment_spec.bento_name)
        _init_azure_functions_project(
            azure_functions_project_dir, bento_path, azure_functions_config,
        )
        _call_az_cli(
            command=[
                'az',
                'group',
                'create',
                '--name',
                resource_group_name,
                '--location',
                azure_functions_config.location,
            ],
            message='create Azure resource group',
        )
        _call_az_cli(
            command=[
                'az',
                'storage',
                'account',
                'create',
                '--name',
                storage_account_name,
                '--resource-group',
                resource_group_name,
            ],
            message='Create Azure storage account',
        )
        _call_az_cli(
            command=[
                'az',
                'functionapp',
                'plan',
                'create',
                '--name',
                function_plan_name,
                '--is-linux',
                '--sku',
                azure_functions_config.premium_plan_sku or DEFAULT_PREMIUM_PLAN_SKU,
                '--min-instances',
                str(azure_functions_config.min_instances)
                or str(DEFAULT_MIN_INSTANCE_COUNT),
                '--max-burst',
                str(azure_functions_config.max_burst) or str(DEFAULT_MAX_BURST),
                '--resource-group',
                resource_group_name,
            ],
            message='create Azure functionapp premium plan',
        )
        # Add note for why choose standard
        _call_az_cli(
            command=[
                'az',
                'acr',
                'create',
                '--name',
                container_registry_name,
                '--sku',
                'standard',
                '--resource-group',
                resource_group_name,
            ],
            message='create Azure container registry',
        )
        try:
            docker_tag = _build_and_push_docker_image_to_azure_container_registry(
                azure_functions_project_dir=azure_functions_project_dir,
                container_registry_name=container_registry_name,
                resource_group_name=resource_group_name,
                bento_name=bento_pb.name,
                bento_version=bento_pb.version,
                bento_python_version=bento_pb.bento_service_metadata.env.python_version,
            )
        except Exception as e:
            raise AzureServiceError(
                f'Failed to create Azure Function docker image. {str(e)}'
            )
        docker_username, docker_password = _get_docker_login_info(
            resource_group_name, container_registry_name
        )
        _call_az_cli(
            command=[
                'az',
                'functionapp',
                'create',
                '--name',
                function_name,
                '--storage-account',
                storage_account_name,
                '--resource-group',
                resource_group_name,
                '--plan',
                function_plan_name,
                '--functions-version',
                '3',
                '--deployment-container-image-name',
                docker_tag,
                '--docker-registry-server-user',
                docker_username,
                '--docker-registry-server-password',
                docker_password,
            ],
            message='create Azure functionapp',
        )
        _set_cors_settings(function_name, resource_group_name)


def _update_azure_functions(
    namespace, deployment_name, deployment_spec, bento_pb, bento_path,
):
    azure_functions_config = deployment_spec.azure_functions_operator_config
    (
        resource_group_name,
        _,
        _,
        function_name,
        container_registry_name,
    ) = _generate_azure_resource_names(namespace, deployment_name)
    with TempDirectory() as temp_dir:
        azure_functions_project_dir = os.path.join(temp_dir, deployment_spec.bento_name)
        _init_azure_functions_project(
            azure_functions_project_dir, bento_path, azure_functions_config,
        )
        docker_tag = _build_and_push_docker_image_to_azure_container_registry(
            azure_functions_project_dir=azure_functions_project_dir,
            container_registry_name=container_registry_name,
            resource_group_name=resource_group_name,
            bento_name=bento_pb.name,
            bento_version=bento_pb.version,
            bento_python_version=bento_pb.bento_service_metadata.env.python_version,
        )
        _call_az_cli(
            command=[
                'az',
                'functionapp',
                'config',
                'container',
                'set',
                '--name',
                function_name,
                '--resource-group',
                resource_group_name,
                '--docker-custom-image-name',
                docker_tag,
            ],
            message='update Azure functionapp settings',
        )


class AzureFunctionsDeploymentOperator(DeploymentOperatorBase):
    def __init__(self, yatai_service):
        super(AzureFunctionsDeploymentOperator, self).__init__(yatai_service)
        ensure_docker_available_or_raise()
        _assert_azure_cli_available()
        _assert_az_cli_logged_in()

    def add(self, deployment_pb):
        try:
            deployment_spec = deployment_pb.spec
            if not deployment_spec.azure_functions_operator_config.location:
                raise YataiDeploymentException(
                    'Azure Functions parameter "location" is missing'
                )
            bento_repo_pb = self.yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            return self._add(
                deployment_pb, bento_repo_pb.bento, bento_repo_pb.bento.uri.uri
            )
        except BentoMLException as error:
            deployment_pb.state.state = DeploymentState.ERROR
            deployment_pb.state.error_message = f'Error: {str(error)}'
            return ApplyDeploymentResponse(
                status=error.status_proto, deployment=deployment_pb
            )

    def _add(self, deployment_pb, bento_pb, bento_path):
        if loader._is_remote_path(bento_path):
            with loader._resolve_remote_bundle_path(bento_path) as local_path:
                return self._add(deployment_pb, bento_pb, local_path)
        try:
            _deploy_azure_functions(
                deployment_spec=deployment_pb.spec,
                deployment_name=deployment_pb.name,
                namespace=deployment_pb.namespace,
                bento_pb=bento_pb,
                bento_path=bento_path,
            )
            return ApplyDeploymentResponse(status=Status.OK(), deployment=deployment_pb)
        except AzureServiceError as error:
            resource_group_name, _, _, _, _, = _generate_azure_resource_names(
                deployment_pb.namespace, deployment_pb.name
            )
            logger.debug(
                'Failed to create Azure Functions. Cleaning up Azure resources'
            )
            try:
                _call_az_cli(
                    command=[
                        'az',
                        'group',
                        'delete',
                        '-y',
                        '--name',
                        resource_group_name,
                    ],
                    message='delete Azure resource group',
                )
            except AzureServiceError:
                pass
            raise error

    def update(self, deployment_pb, previous_deployment):
        try:
            bento_repo_pb = self.yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_pb.spec.bento_name,
                    bento_version=deployment_pb.spec.bento_version,
                )
            )
            bento_pb = bento_repo_pb.bento
            return self._update(
                deployment_pb, previous_deployment, bento_pb, bento_pb.uri.uri
            )
        except BentoMLException as error:
            deployment_pb.state.state = DeploymentState.ERROR
            deployment_pb.state.error_message = (
                f'Encounter error when updating Azure Functions deployment: '
                f'{str(error)}'
            )
            return ApplyDeploymentResponse(
                status=error.status_proto, deployment=deployment_pb
            )

    def _update(self, deployment_pb, current_deployment, bento_pb, bento_path):
        if loader._is_remote_path(bento_path):
            with loader._resolve_remote_bundle_path(bento_path) as local_path:
                return self._update(
                    deployment_pb, current_deployment, bento_pb, local_path
                )
        if (
            deployment_pb.spec.bento_name != current_deployment.spec.bento_name
            or deployment_pb.spec.bento_version != current_deployment.spec.bento_version
        ):
            logger.debug(
                'BentoService tag is different from current Azure Functions '
                'deployment, creating new Azure Functions project and push to ACR'
            )
            _update_azure_functions(
                deployment_spec=deployment_pb.spec,
                deployment_name=deployment_pb.name,
                namespace=deployment_pb.namespace,
                bento_pb=bento_pb,
                bento_path=bento_path,
            )
        (
            resource_group_name,
            _,
            function_plan_name,
            _,
            _,
        ) = _generate_azure_resource_names(
            namespace=deployment_pb.namespace, deployment_name=deployment_pb.name
        )
        _call_az_cli(
            command=[
                'az',
                'functionapp',
                'plan',
                'update',
                '--name',
                function_plan_name,
                '--resource-group',
                resource_group_name,
                '--max-burst',
                str(deployment_pb.spec.azure_functions_operator_config.max_burst),
                '--min-instances',
                str(deployment_pb.spec.azure_functions_operator_config.min_instances),
                '--sku',
                deployment_pb.spec.azure_functions_operator_config.premium_plan_sku,
            ],
            message='update Azure functionapp plan',
        )
        return ApplyDeploymentResponse(deployment=deployment_pb, status=Status.OK())

    def delete(self, deployment_pb):
        try:
            resource_group_name, _, _, _, _ = _generate_azure_resource_names(
                namespace=deployment_pb.namespace, deployment_name=deployment_pb.name
            )
            _call_az_cli(
                command=['az', 'group', 'delete', '-y', '--name', resource_group_name],
                message='delete Azure resource group',
                parse_json=False,
            )
            return DeleteDeploymentResponse(status=Status.OK())
        except BentoMLException as error:
            return DeleteDeploymentResponse(status=error.status_proto)

    def describe(self, deployment_pb):
        try:
            (
                resource_group_name,
                _,
                _,
                function_name,
                _,
            ) = _generate_azure_resource_names(
                deployment_pb.namespace, deployment_pb.name
            )
            show_function_result = _call_az_cli(
                command=[
                    'az',
                    'functionapp',
                    'show',
                    '--name',
                    function_name,
                    '--resource-group',
                    resource_group_name,
                ],
                message='show Azure functionapp detail',
            )
            keys = [
                'defaultHostName',
                'enabledHostNames',
                'hostNames',
                'id',
                'kind',
                'lastModifiedTimeUtc',
                'location',
                'name',
                'repositorySiteName',
                'reserved',
                'resourceGroup',
                'state',
                'type',
                'usageState',
            ]
            # Need find more documentation on the status of functionapp. For now, any
            # other status is error.
            if show_function_result['state'] == 'Running':
                state = DeploymentState.RUNNING
            else:
                state = DeploymentState.ERROR
            info_json = {k: v for k, v in show_function_result.items() if k in keys}
            deployment_state = DeploymentState(
                info_json=json.dumps(info_json, default=str), state=state,
            )
            deployment_state.timestamp.GetCurrentTime()
            return DescribeDeploymentResponse(
                state=deployment_state, status=Status.OK()
            )
        except BentoMLException as error:
            return DescribeDeploymentResponse(status=error.status_proto)
