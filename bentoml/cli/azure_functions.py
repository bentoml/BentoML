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
from datetime import datetime

import click

from bentoml.cli.click_utils import (
    BentoMLCommandGroup,
    parse_bento_tag_callback,
    CLI_COLOR_SUCCESS,
    CLI_COLOR_ERROR,
    _echo,
    parse_labels_callback,
    validate_labels_query_callback,
)
from bentoml.cli.deployment import _print_deployment_info, _print_deployments_info
from bentoml.cli.utils import Spinner
from bentoml.yatai.deployment.azure_functions.operator import (
    AZURE_FUNCTIONS_PREMIUM_PLAN_SKUS,
    AZURE_FUNCTIONS_AUTH_LEVELS,
    DEFAULT_MIN_INSTANCE_COUNT,
    DEFAULT_MAX_BURST,
    DEFAULT_PREMIUM_PLAN_SKU,
    DEFAULT_FUNCTION_AUTH_LEVEL,
)
from bentoml.yatai.deployment.store import ALL_NAMESPACE_TAG
from bentoml.exceptions import BentoMLException
from bentoml.yatai.proto import status_pb2
from bentoml.yatai.proto.deployment_pb2 import DeploymentSpec
from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.utils.usage_stats import track_cli
from bentoml.yatai.client import YataiClient

PLATFORM_NAME = DeploymentSpec.DeploymentOperator.Name(DeploymentSpec.AZURE_FUNCTIONS)


def get_azure_functions_sub_command():
    # pylint: disable=unused-variable

    @click.group(
        name='azure-functions',
        help='Commands for Azure Functions BentoService deployment',
        cls=BentoMLCommandGroup,
    )
    def azure_functions():
        pass

    @azure_functions.command(help='Deploy BentoService to Azure Functions')
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration yatai_service/default_namespace',
    )
    @click.argument('name', type=click.STRING)
    @click.option(
        '-b',
        '--bento',
        '--bento-service-bundle',
        type=click.STRING,
        required=True,
        callback=parse_bento_tag_callback,
        help='Target BentoService to be deployed, referenced by its name and version '
        'in the format of name:version. For example: "iris_classifier:v1.2.0"',
    )
    @click.option(
        '--location',
        type=click.STRING,
        help='The Azure location name for the deployment',
        required=True,
    )
    @click.option(
        '--min-instances',
        type=click.INT,
        default=DEFAULT_MIN_INSTANCE_COUNT,
        help=f'The minimum number of workers for the deployment. The default value is '
        f'{DEFAULT_MIN_INSTANCE_COUNT}',
    )
    @click.option(
        '--max-burst',
        type=click.INT,
        default=DEFAULT_MAX_BURST,
        help=f'The maximum number of elastic workers for the deployment. '
        f'The default value is {DEFAULT_MAX_BURST}',
    )
    @click.option(
        '--premium-plan-sku',
        type=click.Choice(AZURE_FUNCTIONS_PREMIUM_PLAN_SKUS),
        default=DEFAULT_PREMIUM_PLAN_SKU,
        help=f'The Azure Functions premium SKU for the deployment. The default value '
        f'is {DEFAULT_PREMIUM_PLAN_SKU}',
    )
    @click.option(
        '-l',
        '--labels',
        type=click.STRING,
        callback=parse_labels_callback,
        help='Key:value pairs that are attached to deployments and intended to be used '
        'to specify identifying attributes of the deployments that are meaningful to '
        'users. Multiple labels are separated with `,`',
    )
    @click.option(
        '--function-auth-level',
        type=click.Choice(AZURE_FUNCTIONS_AUTH_LEVELS),
        default=DEFAULT_FUNCTION_AUTH_LEVEL,
        help=f'The authorization level for the deployed Azure Functions. The default '
        f'value is {DEFAULT_FUNCTION_AUTH_LEVEL}',
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for apply action to complete or encounter an error.'
        'If set to no-wait, CLI will exit without waiting until it has verified '
        'cloud resource allocation',
    )
    def deploy(
        namespace,
        name,
        bento,
        location,
        min_instances,
        max_burst,
        premium_plan_sku,
        labels,
        function_auth_level,
        output,
        wait,
    ):
        track_cli('deploy-create', PLATFORM_NAME)
        bento_name, bento_version = bento.split(':')
        yatai_client = YataiClient()
        try:
            with Spinner(f'Deploying {bento} to Azure Functions'):
                result = yatai_client.deployment.create_azure_functions_deployment(
                    name=name,
                    namespace=namespace,
                    labels=labels,
                    bento_name=bento_name,
                    bento_version=bento_version,
                    location=location,
                    min_instances=min_instances,
                    max_burst=max_burst,
                    premium_plan_sku=premium_plan_sku,
                    function_auth_level=function_auth_level,
                    wait=wait,
                )
                if result.status.status_code != status_pb2.Status.OK:
                    error_code, error_message = status_pb_to_error_code_and_message(
                        result.status
                    )
                    _echo(
                        f'Failed to create Azure Functions deployment {name} '
                        f'{error_code}:{error_message}',
                        CLI_COLOR_ERROR,
                    )
                    return
                track_cli('deploy-create-success', PLATFORM_NAME)
                _echo(
                    f'Successfully created Azure Functions deployment {name}',
                    CLI_COLOR_SUCCESS,
                )
                _print_deployment_info(result.deployment, output)
        except BentoMLException as e:
            _echo(
                f'Failed to create Azure Functions deployment {name}. {str(e)}',
                CLI_COLOR_ERROR,
            )

    @azure_functions.command(help='Update existing Azure Functions deployment')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-b',
        '--bento',
        '--bento-service-bundle',
        type=click.STRING,
        callback=parse_bento_tag_callback,
        help='Target BentoService to be deployed, referenced by its name and version '
        'in the format of name:version. For example: "iris_classifier:v1.2.0"',
    )
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, the default value is "dev" '
        'which can be changed in BentoML configuration file',
    )
    @click.option(
        '--min-instances',
        type=click.INT,
        help='The minimum number of workers for the deployment.',
    )
    @click.option(
        '--max-burst',
        type=click.INT,
        help='The maximum number of elastic workers for the deployment.',
    )
    @click.option(
        '--premium-plan-sku',
        type=click.Choice(AZURE_FUNCTIONS_PREMIUM_PLAN_SKUS),
        help='The Azure Functions premium SKU for the deployment.',
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for apply action to complete or encounter an error.'
        'If set to no-wait, CLI will exit without waiting until it has verified '
        'cloud resource allocation',
    )
    def update(
        name, namespace, bento, min_instances, max_burst, premium_plan_sku, output, wait
    ):
        yatai_client = YataiClient()
        track_cli('deploy-update', PLATFORM_NAME)
        if bento:
            bento_name, bento_version = bento.split(':')
        else:
            bento_name = None
            bento_version = None
        try:
            with Spinner(f'Updating Azure Functions deployment {name}'):
                result = yatai_client.deployment.update_azure_functions_deployment(
                    namespace=namespace,
                    deployment_name=name,
                    bento_name=bento_name,
                    bento_version=bento_version,
                    min_instances=min_instances,
                    max_burst=max_burst,
                    premium_plan_sku=premium_plan_sku,
                    wait=wait,
                )
                if result.status.status_code != status_pb2.Status.OK:
                    error_code, error_message = status_pb_to_error_code_and_message(
                        result.status
                    )
                    _echo(
                        f'Failed to update Azure Functions deployment {name}. '
                        f'{error_code}:{error_message}'
                    )
                track_cli('deploy-update-success', PLATFORM_NAME)
                _echo(
                    f'Successfully update Azure Functions deployment {name}',
                    CLI_COLOR_SUCCESS,
                )
                _print_deployment_info(result.deployment, output)
        except BentoMLException as e:
            _echo(
                f'Failed to update Azure Functions deployment {name}: {str(e)}',
                CLI_COLOR_ERROR,
            )

    @azure_functions.command(help='Delete Azure Functions deployment')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration yatai_service/default_namespace',
    )
    @click.option(
        '--force',
        is_flag=True,
        help='force delete the deployment record in database and '
        'ignore errors when deleting cloud resources',
    )
    def delete(name, namespace, force):
        track_cli('deploy-delete', PLATFORM_NAME)
        yatai_client = YataiClient()
        get_deployment_result = yatai_client.deployment.get(namespace, name)
        if get_deployment_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_deployment_result.status
            )
            _echo(
                f'Failed to get Azure Functions deployment {name} for deletion. '
                f'{error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return
        try:
            result = yatai_client.deployment.delete(name, namespace, force)
            if result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    result.status
                )
                _echo(
                    f'Failed to delete Azure Functions deployment {name}. '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            extra_properties = {}
            if get_deployment_result.deployment.created_at:
                stopped_time = datetime.utcnow()
                extra_properties['uptime'] = int(
                    (
                        stopped_time
                        - get_deployment_result.deployment.created_at.ToDatetime()
                    ).total_seconds()
                )
            track_cli('deploy-delete-success', PLATFORM_NAME, extra_properties)
            _echo(
                f'Successfully deleted Azure Functions deployment "{name}"',
                CLI_COLOR_SUCCESS,
            )
        except BentoMLException as e:
            _echo(
                f'Failed to delete Azure Functions deployment {name} {str(e)}',
                CLI_COLOR_ERROR,
            )

    @azure_functions.command(help='Get Azure Functions deployment information')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration yatai_service/default_namespace',
    )
    @click.option(
        '-o', '--output', type=click.Choice(['json', 'yaml', 'table']), default='json'
    )
    def get(name, namespace, output):
        yatai_client = YataiClient()
        track_cli('deploy-get', PLATFORM_NAME)
        try:
            get_result = yatai_client.deployment.get(namespace, name)
            if get_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    get_result.status
                )
                _echo(
                    f'Failed to get Azure Functions deployment {name}. '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            describe_result = yatai_client.deployment.describe(namespace, name)
            if describe_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    describe_result.status
                )
                _echo(
                    f'Failed to retrieve the latest status for Azure Functions '
                    f'deployment {name}. {error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            get_result.deployment.state.CopyFrom(describe_result.state)
            _print_deployment_info(get_result.deployment, output)

        except BentoMLException as e:
            _echo(
                f'Failed to get Azure Functions deployment {name}. {str(e)}',
                CLI_COLOR_ERROR,
            )

    @azure_functions.command(name='list', help='List Azure Functions deployments')
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration yatai_service/default_namespace',
        default=ALL_NAMESPACE_TAG,
    )
    @click.option(
        '--limit',
        type=click.INT,
        help='The maximum amount of Azure Functions deployments to be listed at once',
    )
    @click.option(
        '-l',
        '--labels',
        type=click.STRING,
        callback=validate_labels_query_callback,
        help='List deployments matching the giving labels',
    )
    @click.option(
        '--order-by', type=click.Choice(['created_at', 'name']), default='created_at',
    )
    @click.option(
        '--asc/--desc',
        default=False,
        help='Ascending or descending order for list deployments',
    )
    @click.option(
        '-o', '--output', type=click.Choice(['json', 'yaml', 'table']), default='table'
    )
    def list_deployment(namespace, limit, labels, order_by, asc, output):
        yatai_client = YataiClient()
        track_cli('deploy-list', PLATFORM_NAME)
        try:
            list_result = yatai_client.deployment.list_azure_functions_deployments(
                limit=limit,
                labels_query=labels,
                namespace=namespace,
                order_by=order_by,
                ascending_order=asc,
            )
            if list_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    list_result.status
                )
                _echo(
                    f'Failed to list Azure Functions deployments '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            _print_deployments_info(list_result.deployments, output)
        except BentoMLException as e:
            _echo(
                f'Failed to list Azure Functions deployments {str(e)}', CLI_COLOR_ERROR
            )

    return azure_functions
