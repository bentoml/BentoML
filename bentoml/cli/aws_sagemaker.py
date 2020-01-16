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

from bentoml.cli.utils import status_pb_to_error_code_and_message, Spinner
from bentoml.cli.click_utils import (
    parse_bento_tag_callback,
    CLI_COLOR_ERROR,
    _echo,
    CLI_COLOR_SUCCESS,
)
from bentoml.cli.deployment import (
    _print_deployment_info,
    _print_deployments_info,
)
from bentoml.exceptions import BentoMLException
from bentoml.proto import status_pb2
from bentoml.proto.deployment_pb2 import DeploymentSpec
from bentoml.utils.usage_stats import track_cli
from bentoml.yatai.client import YataiClient

DEFAULT_SAGEMAKER_INSTANCE_TYPE = 'ml.m4.xlarge'
DEFAULT_SAGEMAKER_INSTANCE_COUNT = 1
PLATFORM_NAME = DeploymentSpec.DeploymentOperator.Name(DeploymentSpec.AWS_SAGEMAKER)


def get_aws_sagemaker_sub_command():
    # pylint: disable=unused-variable

    @click.group(
        name='sagemaker',
        help='Commands for creating and managing BentoService deployments on '
        'AWS Sagemaker',
    )
    def aws_sagemaker():
        pass

    @aws_sagemaker.command(help='Deploy BentoService to AWS Sagemaker')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-b',
        '--bento',
        '--bento-service-bundle',
        type=click.STRING,
        required=True,
        callback=parse_bento_tag_callback,
        help='Target BentoService to be deployed, referenced by its name and version '
        'in format of name:version. For example: "iris_classifier:v1.2.0"',
    )
    @click.option(
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
    @click.option(
        '--annotations',
        type=click.STRING,
        help='Used to attach arbitrary metadata to BentoService deployments, BentoML '
        'library and other plugins can then retrieve this metadata.',
    )
    @click.option('--region', help='AWS region name for deployment')
    @click.option(
        '--api-name',
        help='User defined API function will be used for inference.',
        required=True,
    )
    @click.option(
        '--instance-type',
        help='Type of instance will be used for inference. Default to "m1.m4.xlarge"',
        type=click.STRING,
        default=DEFAULT_SAGEMAKER_INSTANCE_TYPE,
    )
    @click.option(
        '--instance-count',
        help='Number of instance will be used. Default value is 1',
        type=click.INT,
        default=DEFAULT_SAGEMAKER_INSTANCE_COUNT,
    )
    @click.option(
        '--num-of-gunicorn-workers-per-instance',
        help='Number of gunicorn worker will be used per instance. Default value for '
        'gunicorn worker is based on the instance\' cpu core counts.  '
        'The formula is num_of_cpu/2 + 1',
        type=click.INT,
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
        bento,
        namespace,
        labels,
        annotations,
        region,
        instance_type,
        instance_count,
        num_of_gunicorn_workers_per_instance,
        api_name,
        output,
        wait,
    ):
        # use the DeploymentOperator name in proto to be consistent with amplitude
        track_cli('deploy-create', PLATFORM_NAME)
        bento_name, bento_version = bento.split(':')
        yatai_client = YataiClient()
        try:
            with Spinner('Deploying Sagemaker deployment '):
                result = yatai_client.deployment.create_sagemaker_deployment(
                    name=name,
                    namespace=namespace,
                    labels=labels,
                    annotations=annotations,
                    bento_name=bento_name,
                    bento_version=bento_version,
                    instance_count=instance_count,
                    instance_type=instance_type,
                    num_of_gunicorn_workers_per_instance=num_of_gunicorn_workers_per_instance,  # noqa E501
                    api_name=api_name,
                    region=region,
                    wait=wait,
                )
            if result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    result.status
                )
                _echo(
                    f'Failed to create Sagemaker deployment {name} '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            track_cli('deploy-create-success', PLATFORM_NAME)
            _echo(
                f'Successfully created Sagemaker deployment {name}', CLI_COLOR_SUCCESS,
            )
            _print_deployment_info(result.deployment, output)
        except BentoMLException as e:
            _echo(
                'Failed to create Sagemaker deployment {}.: {}'.format(name, str(e)),
                CLI_COLOR_ERROR,
            )

    @aws_sagemaker.command(help='Update existing AWS Sagemaker deployment')
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
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        'can be changed in BentoML configuration file',
    )
    @click.option(
        '--instance-type',
        help='Type of instance will be used for inference. Default to "m1.m4.xlarge"',
        type=click.STRING,
    )
    @click.option(
        '--instance-count',
        help='Number of instance will be used. Default value is 1',
        type=click.INT,
    )
    @click.option(
        '--num-of-gunicorn-workers-per-instance',
        help='Number of gunicorn worker will be used per instance. Default value for '
        'gunicorn worker is based on the instance\' cpu core counts.  '
        'The formula is num_of_cpu/2 + 1',
        type=click.INT,
    )
    @click.option(
        '--api-name', help='User defined API function will be used for inference.',
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for apply action to complete or encounter an error.'
        'If set to no-wait, CLI will return immediately. The default value is wait',
    )
    def update(
        name,
        namespace,
        bento,
        api_name,
        instance_type,
        instance_count,
        num_of_gunicorn_workers_per_instance,
        output,
        wait,
    ):
        yatai_client = YataiClient()
        track_cli('deploy-update', PLATFORM_NAME)
        if bento:
            bento_name, bento_version = bento.split(':')
        else:
            bento_name = None
            bento_version = None
        try:
            with Spinner('Updating Sagemaker deployment '):
                result = yatai_client.deployment.update_sagemaker_deployment(
                    namespace=namespace,
                    deployment_name=name,
                    bento_name=bento_name,
                    bento_version=bento_version,
                    instance_count=instance_count,
                    instance_type=instance_type,
                    num_of_gunicorn_workers_per_instance=num_of_gunicorn_workers_per_instance,  # noqa E501
                    api_name=api_name,
                    wait=wait,
                )
            if result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    result.status
                )
                _echo(
                    f'Failed to updated deployment {name}.'
                    f'{error_code}:{error_message}'
                )
            track_cli('deploy-update-success', PLATFORM_NAME)
            _echo(
                f'Successfully updated Sagemaker deployment {name}', CLI_COLOR_SUCCESS
            )
            _print_deployment_info(result.deployment, output)
        except BentoMLException as e:
            _echo(
                f'Failed to update Sagemaker deployment {name}: {str(e)}',
                CLI_COLOR_ERROR,
            )

    @aws_sagemaker.command(help='Delete AWS Sagemaker deployment')
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
        get_deployment_result = yatai_client.deployment.get(namespace, name)
        if get_deployment_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_deployment_result.status
            )
            _echo(
                f'Failed to get Sagemaker deployment {name} for deletion. '
                f'{error_code}:{error_message}',
                CLI_COLOR_ERROR,
            )
            return
        track_cli('deploy-delete', PLATFORM_NAME)
        result = yatai_client.deployment.delete(name, namespace, force)
        if result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            _echo(
                f'Failed to delete Sagemaker deployment {name}. '
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
        _echo(f'Successfully deleted Sagemaker deployment "{name}"', CLI_COLOR_SUCCESS)

    @aws_sagemaker.command(help='Get AWS Sagemaker deployment information')
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
                error_code, error_message = status_pb_to_error_code_and_message(
                    get_result.status
                )
                _echo(
                    f'Failed to get Sagemaker deployment {name}. '
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
                    f'Failed to retrieve the latest status for Sagemaker deployment '
                    f'{name}. {error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
                return
            get_result.deployment.state.CopyFrom(describe_result.state)
            _print_deployment_info(get_result.deployment, output)
            return
        else:
            track_cli('deploy-list', PLATFORM_NAME)
            list_result = yatai_client.deployment.list_sagemaker_deployments(
                limit=limit,
                offset=offset,
                filters=filters,
                labels=labels,
                namespace=namespace,
                is_all_namespaces=all_namespaces,
            )
            if list_result.status.status_code != status_pb2.Status.OK:
                error_code, error_message = status_pb_to_error_code_and_message(
                    list_result.status
                )
                _echo(
                    f'Failed to list Sagemaker deployments. '
                    f'{error_code}:{error_message}',
                    CLI_COLOR_ERROR,
                )
            else:
                _print_deployments_info(list_result.deployments, output)

    return aws_sagemaker
