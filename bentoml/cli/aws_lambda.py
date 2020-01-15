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

from bentoml.cli.utils import parse_pb_response_error_message
from bentoml.cli.click_utils import (
    parse_bento_tag_callback,
    CLI_COLOR_ERROR,
    _echo,
    CLI_COLOR_SUCCESS,
)
from bentoml.cli.deployment import (
    get_state_after_await_action_complete,
    _print_deployment_info,
    _print_deployments_info,
)
from bentoml.exceptions import BentoMLException
from bentoml.proto import status_pb2
from bentoml.proto.deployment_pb2 import DeploymentSpec
from bentoml.utils.usage_stats import track_cli
from bentoml.yatai.client import YataiClient

PLATFORM_NAME = DeploymentSpec.DeploymentOperator.Name(DeploymentSpec.AWS_LAMBDA)


def get_aws_lambda_sub_command():
    # pylint: disable=unused-variable

    @click.group(
        help='Commands for creating and managing BentoService deployments on AWS Lambda'
    )
    def aws_lambda():
        pass

    @aws_lambda.command(help='Deploy BentoService to AWS Lambda')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-b',
        '--bento',
        '--bento-service-bundle',
        type=click.STRING,
        callback=parse_bento_tag_callback,
        help='Target BentoService to be deployed, referenced by its name and version '
        'in format of name:version. For example: "iris_classifier:v1.2.0"',
    )
    @click.option(
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "default" which'
        'can be changed in BentoML configuration file',
    )
    @click.option(
        '-l',
        '--labels',
        type=click.STRING,
        help='Key:value pairs that are attached to deployments and intended to be used'
        'to specify identifying attributes of the deployments that are meaningful to '
        'users',
    )
    @click.option(
        '--annotations',
        type=click.STRING,
        help='Used to attach arbitrary metadata to BentoService deployments, BentoML '
        'library and other plugins can then retrieve this metadata.',
    )
    @click.option('--region', help='AWS region name for deployment')
    @click.option(
        '--api-name',
        help='User defined API function will be used for inference',
    )
    @click.option(
        '--memory-size',
        help="Maximum Memory Capacity for AWS Lambda function, you can set the memory "
        "size in 64MB increments from 128MB to 3008MB. The default value "
        "is 1024 MB.",
        type=click.INT,
        default=1024,
    )
    @click.option(
        '--timeout',
        help="The amount of time that AWS Lambda allows a function to run before "
        "stopping it. The default is 3 seconds. The maximum allowed value is "
        "900 seconds",
        type=click.INT,
        default=3,
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for apply action to complete or encounter an error.'
        'If set to no-wait, CLI will return immediately. The default value is wait',
    )
    def deploy(
        name,
        namespace,
        bento,
        labels,
        annotations,
        region,
        api_name,
        memory_size,
        timeout,
        output,
        wait,
    ):
        track_cli('deploy-create', PLATFORM_NAME)
        yatai_client = YataiClient()
        bento_name, bento_version = bento.split(':')
        try:
            result = yatai_client.deployment.create_lambda_deployment(
                name=name,
                namespace=namespace,
                bento_name=bento_name,
                bento_version=bento_version,
                api_name=api_name,
                region=region,
                memory_size=memory_size,
                timeout=timeout,
                labels=labels,
                annotations=annotations,
            )
        except BentoMLException as e:
            _echo(
                f'Failed to create Lambda deployment {name} {str(e)}', CLI_COLOR_ERROR
            )
        if result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = parse_pb_response_error_message(result.status)
            _echo(
                f'Failed to create Lambda deployment {name} '
                f'{error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return
        if wait:
            result_state = get_state_after_await_action_complete(
                yatai_client=yatai_client,
                name=name,
                namespace=namespace,
                message='Creating Lambda deployment ',
            )
            if result_state.status.status_code != status_pb2.Status.OK:
                error_code, error_message = parse_pb_response_error_message(
                    result_state.status
                )
                _echo(
                    f'Created Lambda deployment {name}, failed to retrieve latest '
                    f'status {error_code}:{error_message}'
                )
            result.deployment.state.CopyFrom(result_state.state)
        track_cli('deploy-create-success', PLATFORM_NAME)
        _echo(f'Successfully created Lambda deployment {name}', CLI_COLOR_SUCCESS)
        _print_deployment_info(result.deployment, output)

    @aws_lambda.command(help='Update existing AWS Lambda deployment')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-b',
        '--bento',
        '--bento-service-bundle',
        type=click.STRING,
        callback=parse_bento_tag_callback,
        help='Target BentoService to be deployed, referenced by its name and version '
        'in format of name:version. For example: "iris_classifier:v1.2.0"',
    )
    @click.option(
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "default" which'
        'can be changed in BentoML configuration file',
    )
    @click.option(
        '--api-name',
        help='User defined API function will be used for inference.',
    )
    @click.option(
        '--memory-size',
        help="Maximum Memory Capacity for AWS Lambda function, you can set the memory "
        "size in 64MB increments from 128MB to 3008MB. The default value "
        "is 1024 MB.",
        type=click.INT,
    )
    @click.option(
        '--timeout',
        help="The amount of time that AWS Lambda allows a function to run before "
        "stopping it. The default is 3 seconds. The maximum allowed value is "
        "900 seconds",
        type=click.INT,
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for apply action to complete or encounter an error.'
        'If set to no-wait, CLI will return immediately. The default value is wait',
    )
    def update(name, namespace, bento, api_name, memory_size, timeout, output, wait):
        yatai_client = YataiClient()
        if bento:
            bento_name, bento_version = bento.split(':')
        else:
            bento_name = None
            bento_version = None
        try:
            result = yatai_client.deployment.update_lambda_deployment(
                bento_name=bento_name,
                bento_version=bento_version,
                deployment_name=name,
                namespace=namespace,
                api_name=api_name,
                memory_size=memory_size,
                timeout=timeout,
            )
        except BentoMLException as e:
            _echo(
                f'Failed to updated Lambda deployment {name}: {str(e)}', CLI_COLOR_ERROR
            )
            return
        if result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = parse_pb_response_error_message(result.status)
            _echo(
                f'Failed to update Lambda deployment {name} '
                f'{error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return
        if wait:
            result_state = get_state_after_await_action_complete(
                yatai_client=yatai_client,
                name=name,
                namespace=namespace,
                message='Updating Lambda deployment ',
            )
            if result_state.status.status_code != status_pb2.Status.OK:
                error_code, error_message = parse_pb_response_error_message(
                    result_state.status
                )
                _echo(
                    f'Updated Lambda deployment {name}. Failed to retrieve latest '
                    f'status {error_code}:{error_message}'
                )
                return
            result.deployment.state.CopyFrom(result_state.state)
        track_cli('deploy-update-success', PLATFORM_NAME)
        _echo(f'Successfully updated Lambda deployment {name}', CLI_COLOR_SUCCESS)
        _print_deployment_info(result.deployment, output)

    @aws_lambda.command(help='Delete AWS Lambda deployment')
    @click.argument('name', type=click.STRING)
    @click.option(
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration file',
    )
    @click.option(
        '--force',
        is_flag=True,
        help='force delete the deployment record in database and '
        'ignore errors when deleting cloud resources',
    )
    def delete(name, namespace, force):
        yatai_client = YataiClient()
        get_deployment_result = yatai_client.deployment.get(
            namespace=namespace, name=name
        )
        if get_deployment_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = parse_pb_response_error_message(
                get_deployment_result.status
            )
            _echo(
                f'Failed to get Lambda deployment {name} for deletion '
                f'{error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return
        track_cli('deploy-delete', PLATFORM_NAME)
        try:
            result = yatai_client.deployment.delete(
                namespace=namespace, deployment_name=name, force_delete=force
            )
            if result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = parse_pb_response_error_message(
                    result.status
                )
                _echo(
                    f'Failed to delete Lambda deployment {name} '
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
            _echo(f'Successfully deleted Lambda deployment "{name}"', CLI_COLOR_SUCCESS)
        except BentoMLException as e:
            _echo(
                f'Failed to delete Lambda deployment {name} {str(e)}', CLI_COLOR_ERROR
            )

    @aws_lambda.command(help='Get AWS Lambda deployment information')
    @click.option('-n', '--name', type=click.STRING, help='Deployment name')
    @click.option(
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration file',
    )
    @click.option('--all-namespaces', is_flag=True)
    @click.option(
        '--limit', type=click.INT, help='Limit how many deployments will be retrieved'
    )
    @click.option(
        '--offset',
        type=click.INT,
        help='Offset number of deployments will be retrieved',
    )
    @click.option(
        '--filters',
        type=click.STRING,
        help='List deployments containing the filter string in name or version',
    )
    @click.option(
        '-l',
        '--labels',
        type=click.STRING,
        help='List deployments matching the giving labels',
    )
    @click.option(
        '-o', '--output', type=click.Choice(['json', 'yaml', 'table']), default='table'
    )
    def get(name, namespace, all_namespaces, limit, offset, filters, labels, output):
        yatai_client = YataiClient()
        if name:
            track_cli('deploy-get', PLATFORM_NAME)
            get_result = yatai_client.deployment.get(namespace, name)
            if get_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = parse_pb_response_error_message(
                    get_result.status
                )
                _echo(
                    f'Failed to get Lambda deployment {name}. '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            describe_result = yatai_client.deployment.describe(namespace, name)
            if describe_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = parse_pb_response_error_message(
                    describe_result.status
                )
                _echo(
                    f'Failed to retrieve the latest status for Lambda deployment '
                    f'{name}. {error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            get_result.deployment.state.CopyFrom(describe_result.state)
            _print_deployment_info(get_result.deployment, output)
            return
        else:
            track_cli('deploy-list', PLATFORM_NAME)
            list_result = yatai_client.deployment.list_lambda_deployments(
                limit=limit,
                offset=offset,
                filters=filters,
                labels=labels,
                namespace=namespace,
                is_all_namespaces=all_namespaces,
            )
            if list_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = parse_pb_response_error_message(
                    list_result.status
                )
                _echo(
                    f'Failed to list Sagemaker deployments. '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
            else:
                _print_deployments_info(list_result.deployments, output)

    return aws_lambda
