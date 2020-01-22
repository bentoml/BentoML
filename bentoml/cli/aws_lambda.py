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

from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.cli.utils import Spinner
from bentoml.cli.click_utils import (
    BentoMLCommandGroup,
    parse_bento_tag_callback,
    CLI_COLOR_ERROR,
    _echo,
    CLI_COLOR_SUCCESS,
)
from bentoml.cli.deployment import (
    _print_deployment_info,
    _print_deployments_info,
)
from bentoml.deployment.store import ALL_NAMESPACE_TAG
from bentoml.exceptions import BentoMLException
from bentoml.proto import status_pb2
from bentoml.proto.deployment_pb2 import DeploymentSpec
from bentoml.utils.usage_stats import track_cli
from bentoml.yatai.client import YataiClient

PLATFORM_NAME = DeploymentSpec.DeploymentOperator.Name(DeploymentSpec.AWS_LAMBDA)


def get_aws_lambda_sub_command():
    # pylint: disable=unused-variable

    @click.group(
        name='lambda',
        help='Commands for AWS Lambda BentoService deployments',
        cls=BentoMLCommandGroup,
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
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
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
    @click.option('--region', help='AWS region name for deployment')
    @click.option(
        '--api-name', help='User defined API function will be used for inference',
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
            with Spinner('Deploying Lambda deployment '):
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
                    wait=wait,
                )
            if result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    result.status
                )
                track_cli(
                    'deploy-create-failure',
                    PLATFORM_NAME,
                    {'error_code': error_code, 'error_message': error_message},
                )
                _echo(
                    f'Failed to create AWS Lambda deployment {name} '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            track_cli('deploy-create-success', PLATFORM_NAME)
            _echo(
                f'Successfully created AWS Lambda deployment {name}', CLI_COLOR_SUCCESS
            )
            _print_deployment_info(result.deployment, output)
        except BentoMLException as e:
            track_cli('deploy-create-failure', PLATFORM_NAME, {'error_message': str(e)})
            _echo(
                f'Failed to create AWS Lambda deployment {name} {str(e)}',
                CLI_COLOR_ERROR,
            )

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
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration file',
    )
    @click.option(
        '--memory-size',
        help="Maximum memory capacity for AWS Lambda function in MB, you can set "
        "the memory size in 64MB increments from 128 to 3008. "
        "The default value is 1024",
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
    def update(name, namespace, bento, memory_size, timeout, output, wait):
        yatai_client = YataiClient()
        if bento:
            bento_name, bento_version = bento.split(':')
        else:
            bento_name = None
            bento_version = None
        try:
            with Spinner('Updating Lambda deployment '):
                result = yatai_client.deployment.update_lambda_deployment(
                    bento_name=bento_name,
                    bento_version=bento_version,
                    deployment_name=name,
                    namespace=namespace,
                    memory_size=memory_size,
                    timeout=timeout,
                    wait=wait,
                )
                if result.status.status_code != status_pb2.Status.OK:
                    error_code, error_message = status_pb_to_error_code_and_message(
                        result.status
                    )
                    track_cli(
                        'deploy-update-failure',
                        PLATFORM_NAME,
                        {'error_code': error_code, 'error_message': error_message},
                    )
                    _echo(
                        f'Failed to update AWS Lambda deployment {name} '
                        f'{error_code}:{error_message}',
                        CLI_COLOR_ERROR,
                    )
                    return
                track_cli('deploy-update-success', PLATFORM_NAME)
                _echo(
                    f'Successfully updated AWS Lambda deployment {name}',
                    CLI_COLOR_SUCCESS,
                )
                _print_deployment_info(result.deployment, output)
        except BentoMLException as e:
            track_cli('deploy-update-failure', PLATFORM_NAME, {'error_message': str(e)})
            _echo(
                f'Failed to updated AWS Lambda deployment {name}: {str(e)}',
                CLI_COLOR_ERROR,
            )

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
            error_code, error_message = status_pb_to_error_code_and_message(
                get_deployment_result.status
            )
            _echo(
                f'Failed to get AWS Lambda deployment {name} for deletion '
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
                error_code, error_message = status_pb_to_error_code_and_message(
                    result.status
                )
                track_cli(
                    'deploy-delete-failure',
                    PLATFORM_NAME,
                    {'error_code': error_code, 'error_message': error_message},
                )
                _echo(
                    f'Failed to delete AWS Lambda deployment {name} '
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
                f'Successfully deleted AWS Lambda deployment "{name}"',
                CLI_COLOR_SUCCESS,
            )
        except BentoMLException as e:
            track_cli('deploy-delete-failure', PLATFORM_NAME, {'error_message': str(e)})
            _echo(
                f'Failed to delete AWS Lambda deployment {name} {str(e)}',
                CLI_COLOR_ERROR,
            )

    @aws_lambda.command(help='Get AWS Lambda deployment information')
    @click.argument('name', type=click.STRING)
    @click.option(
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration file',
    )
    @click.option(
        '-o', '--output', type=click.Choice(['json', 'yaml', 'table']), default='json'
    )
    def get(name, namespace, output):
        yatai_client = YataiClient()
        track_cli('deploy-get', PLATFORM_NAME)
        describe_result = yatai_client.deployment.describe(namespace, name)
        if describe_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                describe_result.status
            )
            _echo(
                f'Failed to retrieve the latest status for AWS Lambda deployment '
                f'{name}. {error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return

        get_result = yatai_client.deployment.get(namespace, name)
        if get_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_result.status
            )
            _echo(
                f'Failed to get AWS Lambda deployment {name}. '
                f'{error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return
        _print_deployment_info(get_result.deployment, output)

    @aws_lambda.command(name='list', help='List AWS Sagemaker deployments')
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration file',
        default=ALL_NAMESPACE_TAG,
    )
    @click.option(
        '--limit',
        type=click.INT,
        help='The maximum amount of AWS Lambda deployments to be listed at once',
    )
    @click.option(
        '-l',
        '--labels',
        type=click.STRING,
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
    def list_deployments(namespace, limit, labels, order_by, asc, output):
        yatai_client = YataiClient()
        track_cli('deploy-list', PLATFORM_NAME)
        try:
            list_result = yatai_client.deployment.list_lambda_deployments(
                limit=limit,
                labels=labels,
                namespace=namespace,
                order_by=order_by,
                ascending_order=asc,
            )
            if list_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    list_result.status
                )
                _echo(
                    f'Failed to list AWS Sagemaker deployments. '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
            else:
                _print_deployments_info(list_result.deployments, output)
        except BentoMLException as e:
            _echo(f'Failed to list AWS Sagemaker deployment {str(e)}')

    return aws_lambda
