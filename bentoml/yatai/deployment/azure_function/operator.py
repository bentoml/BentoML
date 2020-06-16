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

from bentoml import config
from bentoml.utils.tempdir import TempDirectory
from bentoml.saved_bundle import loader
from bentoml.yatai.deployment.azure_function.templates import (
    AZURE_FUNCTION_HOST_JSON,
    AZURE_FUNCTION_LOCAL_SETTING_JSON,
    AZURE_API_INIT_PY,
    AZURE_API_FUNCTION_JSON,
    BENTO_SERVICE_AZURE_FUNCTION_DOCKERFILE,
)
from bentoml.yatai.deployment.operator import DeploymentOperatorBase
from bentoml.yatai.deployment.utils import ensure_docker_available_or_raise
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


def ensure_azure_cli_available_and_login_or_raise():
    try:
        account_info = _call_az_cli(
            command=['az', 'account', 'show'], message='show Azure account'
        )
        if not account_info['user'] or not account_info['homeTenantId']:
            # https://docs.microsoft.com/en-us/cli/azure/create-an-azure-service-principal-azure-cli?view=azure-cli-latest
            # Attempt to login with azure service principal values from bentoml
            # configuration.
            service_principal_app_id = config.get('yatai_service').get(
                'azure_service_principal_app_url', None
            )
            service_principal_cert = config('yatai_service').get(
                'azure_service_principal_cert', None
            )
            service_principal_tenant = config('yatai_service').get(
                'azure_service_principal_tenant', None
            )
            if (
                service_principal_app_id
                and service_principal_cert
                and service_principal_tenant
            ):
                service_principal_login_result = _call_az_cli(
                    command=[
                        'az',
                        'login',
                        '--service-principal',
                        '--user-name',
                        service_principal_app_id,
                        '--password',
                        service_principal_cert,
                        '--tenant',
                        service_principal_tenant,
                    ],
                    message='login Azure CLI with service principal account',
                )
                if service_principal_login_result['state'] != 'Enabled':
                    raise BentoMLException(
                        'Azure service principal account is not enabled'
                    )
            else:
                raise BentoMLException('Azure CLI is not logged in')
    except FileNotFoundError:
        raise MissingDependencyException(
            'azure cli is required for this deployment. Please visit '
            'https://docs.microsoft.com/en-us/cli/azure/install-azure-cli '
            'for instructions'
        )


def _init_azure_function_project(
    azure_function_project_dir, bento_path, bento_pb, azure_function_config
):
    try:
        shutil.copytree(bento_path, azure_function_project_dir)
        with open(os.path.join(azure_function_project_dir, 'host.json'), 'w') as f:
            f.write(AZURE_FUNCTION_HOST_JSON)
        with open(
            os.path.join(azure_function_project_dir, 'local.settings.json'), 'w'
        ) as f:
            f.write(AZURE_FUNCTION_LOCAL_SETTING_JSON)
        with open(
            os.path.join(azure_function_project_dir, 'Dockerfile-azure'), 'w'
        ) as f:
            f.write(
                BENTO_SERVICE_AZURE_FUNCTION_DOCKERFILE.format(
                    bentoml_version=LAST_PYPI_RELEASE_VERSION
                )
            )

        app_path = os.path.join(azure_function_project_dir, 'app')
        os.mkdir(app_path)
        with open(os.path.join(app_path, '__init__.py'), 'w') as f:
            f.write(AZURE_API_INIT_PY.format(bento_name=bento_pb.name))
        with open(os.path.join(app_path, 'function.json'), 'w') as f:
            f.write(
                AZURE_API_FUNCTION_JSON.format(
                    function_auth_level=azure_function_config.function_auth_level
                )
            )
    except Exception as e:
        raise BentoMLException(f'Failed to initialize azure function project. {str(e)}')


def _generate_azure_resource_names(namespace, deployment_name):
    # Genrate resource names base on
    # https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/resource-name-rules

    # 1-90, alphannumeric(A-Za-z0-9) underscores, parenthese, hyphens, periods
    # scope subscription
    resource_group_name = f'{namespace}-{deployment_name}'
    if len(resource_group_name) > 90:
        resource_group_name = f'{namespace[:30]}-{deployment_name[:60]}'

    # 3-24 a-z0-9, scope: global
    storage_account_name = f'{namespace}{deployment_name}'.lower()
    if len(storage_account_name):
        storage_account_name = f'{namespace[:5]}{deployment_name[0:19]}'
    invalid_chars_for_storage_account = re.compile("[^a-z0-9]")
    storage_account_name = re.sub(
        invalid_chars_for_storage_account, '0', storage_account_name
    )

    # cant find restriction for functionapp plan name.
    function_plan_name = f'{deployment_name}'

    # same as Microsoft.Web/sites
    # 2-60, alphanumeric and hyphens. scope global
    function_name = f'{namespace}-{deployment_name}'
    if len(deployment_name) > 60:
        function_name = f'{namespace[:19]}-{deployment_name[:40]}'
    invalid_chars_for_function_name = re.compile("[^a-zA-Z0-9-]")
    function_name = re.sub(invalid_chars_for_function_name, '-', function_name)

    # 5-50, alphanumeric scope global
    container_registry_name = f'{namespace}{deployment_name}'
    if len(container_registry_name) > 50:
        container_registry_name = f'{namespace[:10]}{deployment_name[:40]}'
    invalid_chars_for_container_registry = re.compile("[^a-zA-Z0-9]")
    container_registry_name = re.sub(
        invalid_chars_for_container_registry, '0', container_registry_name
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
    azure_function_project_dir,
    container_registry_name,
    resource_group_name,
    bento_name,
    bento_version,
):
    _login_acr_registry(container_registry_name, resource_group_name)
    docker_client = docker.from_env()
    tag = f'{container_registry_name}.azurecr.io/{bento_name}:{bento_version}'.lower()
    logger.debug(f'Building docker image {tag}')
    try:
        docker_client.images.build(
            path=azure_function_project_dir,
            dockerfile=os.path.join(azure_function_project_dir, 'Dockerfile-azure'),
            tag=tag,
        )
        logger.debug('Finished building docker image')
    except docker.errors.BuildError as e:
        raise BentoMLException(f'Failed to build docker image. BuildError: {str(e)}')
    except docker.errors.APIError as e:
        raise BentoMLException(f'Failed to build docker image. APIError: {str(e)}')
    try:
        docker_client.images.push(tag)
        logger.debug('Finished pushing docker image')
    except docker.errors.APIError as e:
        raise BentoMLException(f'Failed to push docker image. APIError: {str(e)}')

    return tag


def _get_storage_account_connect_string(resource_group_name, storage_account_name):
    try:
        return _call_az_cli(
            command=[
                'az',
                'storage',
                'account',
                'show-connection-string',
                '--resource-group',
                resource_group_name,
                '--name',
                storage_account_name,
                '--query',
                'connectionString',
                '--output',
                'tsv',
            ],
            message='get Azure storage account connection string',
            parse_json=False,
        )
    except BentoMLException as e:
        raise AzureServiceError(
            f'Failed to get Azure storage account connection string. {str(e)}'
        )


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


def _deploy_azure_function(
    namespace, deployment_name, deployment_spec, bento_pb, bento_path,
):
    azure_function_config = deployment_spec.azure_function_operator_config
    (
        resource_group_name,
        storage_account_name,
        function_plan_name,
        function_name,
        container_registry_name,
    ) = _generate_azure_resource_names(namespace, deployment_name)
    with TempDirectory() as temp_dir:
        azure_function_project_dir = os.path.join(temp_dir, deployment_spec.bento_name)
        _init_azure_function_project(
            azure_function_project_dir, bento_path, bento_pb, azure_function_config,
        )
        _call_az_cli(
            command=[
                'az',
                'group',
                'create',
                '--name',
                resource_group_name,
                '--location',
                azure_function_config.location,
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
                azure_function_config.premium_plan_sku,
                '--min-instances',
                str(azure_function_config.min_instances),
                '--max-burst',
                str(azure_function_config.max_burst),
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
        docker_tag = _build_and_push_docker_image_to_azure_container_registry(
            azure_function_project_dir=azure_function_project_dir,
            container_registry_name=container_registry_name,
            resource_group_name=resource_group_name,
            bento_name=bento_pb.name,
            bento_version=bento_pb.version,
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


def _update_azure_function(
    namespace, deployment_name, deployment_spec, bento_pb, bento_path,
):
    azure_function_config = deployment_spec.azure_function_operator_config
    (
        resource_group_name,
        _,
        _,
        function_name,
        container_registry_name,
    ) = _generate_azure_resource_names(namespace, deployment_name)
    with TempDirectory() as temp_dir:
        azure_function_project_dir = os.path.join(temp_dir, deployment_spec.bento_name)
        _init_azure_function_project(
            azure_function_project_dir, bento_path, bento_pb, azure_function_config,
        )
        docker_tag = _build_and_push_docker_image_to_azure_container_registry(
            azure_function_project_dir=azure_function_project_dir,
            container_registry_name=container_registry_name,
            resource_group_name=resource_group_name,
            bento_name=bento_pb.name,
            bento_version=bento_pb.version,
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


class AzureFunctionDeploymentOperator(DeploymentOperatorBase):
    def __init__(self, yatai_service):
        super(AzureFunctionDeploymentOperator, self).__init__(yatai_service)
        ensure_docker_available_or_raise()
        ensure_azure_cli_available_and_login_or_raise()

    def add(self, deployment_pb):
        try:
            deployment_spec = deployment_pb.spec
            if not deployment_spec.azure_function_operator_config.location:
                raise YataiDeploymentException(
                    'Azure function parameter "location" is missing'
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
            _deploy_azure_function(
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
            logger.debug('Failed to create Azure function. Cleaning up Azure resources')
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
                f'Encounter error when updating Azure function deployment: {str(error)}'
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
                'BentoService tag is different from current Azure function deployment, '
                'creating new Azure function project and push to ACR'
            )
            _update_azure_function(
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
                str(deployment_pb.spec.azure_function_operator_config.max_burst),
                '--min-instances',
                str(deployment_pb.spec.azure_function_operator_config.min_instances),
                '--sku',
                deployment_pb.spec.azure_function_operator_config.premium_plan_sku,
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
