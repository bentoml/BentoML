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

import click

from bentoml.utils.lazy_loader import LazyLoader
from bentoml.cli.utils import Spinner
from bentoml.utils import get_default_yatai_client
from bentoml.cli.click_utils import (
    BentoMLCommandGroup,
    parse_bento_tag_callback,
    _echo,
    CLI_COLOR_SUCCESS,
    parse_labels_callback,
)
from bentoml.cli.deployment import (
    _print_deployment_info,
    _print_deployments_info,
)
from bentoml.yatai.deployment import ALL_NAMESPACE_TAG
from bentoml.exceptions import CLIException
from bentoml.utils import status_pb_to_error_code_and_message

yatai_proto = LazyLoader('yatai_proto', globals(), 'bentoml.yatai.proto')


DEFAULT_SAGEMAKER_INSTANCE_TYPE = 'ml.m4.xlarge'
DEFAULT_SAGEMAKER_INSTANCE_COUNT = 1


def get_aws_sagemaker_sub_command():
    # pylint: disable=unused-variable

    @click.group(
        name='sagemaker',
        help='Commands for AWS Sagemaker BentoService deployments',
        cls=BentoMLCommandGroup,
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
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which '
        'can be changed in BentoML configuration yatai_service/default_namespace',
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
    @click.option(
        '--timeout',
        help="The amount of time Sagemaker will wait before return response",
        type=click.INT,
        default=60,
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for apply action to complete or encounter an error.'
        'If set to no-wait, CLI will return immediately. The default value is wait',
    )
    @click.option(
        '--data-capture-s3-prefix',
        help="To enable data capture (input and output), "
        "provide a destination s3 prefix "
        "(of the format `s3://bucket-name/optional/prefix`)"
        " for the captured data. To disable data capture, leave this blank.",
        type=click.STRING,
        default=None,
    )
    @click.option(
        '--data-capture-sample-percent',
        help="When data capture is enabled, the sampling percentage. Default 100%. "
        "No effect without data-capture-s3-prefix.",
        type=click.IntRange(1, 100),
        default=100,
    )
    def deploy(
        name,
        bento,
        namespace,
        labels,
        region,
        instance_type,
        instance_count,
        num_of_gunicorn_workers_per_instance,
        api_name,
        timeout,
        output,
        wait,
        data_capture_s3_prefix,
        data_capture_sample_percent,
    ):
        _echo(
            message='AWS Sagemaker deployment functionalities are being migrated to a '
            'separate tool and related CLI commands will be deprecated in BentoML '
            'itself, please use https://github.com/bentoml/aws-sagemaker-deploy '
            'going forward.',
            color='yellow',
        )
        # use the DeploymentOperator name in proto to be consistent with amplitude
        bento_name, bento_version = bento.split(':')
        yatai_client = get_default_yatai_client()
        with Spinner('Deploying Sagemaker deployment '):
            result = yatai_client.deployment.create_sagemaker_deployment(
                name=name,
                namespace=namespace,
                labels=labels,
                bento_name=bento_name,
                bento_version=bento_version,
                instance_count=instance_count,
                instance_type=instance_type,
                num_of_gunicorn_workers_per_instance=num_of_gunicorn_workers_per_instance,  # noqa E501
                api_name=api_name,
                timeout=timeout,
                region=region,
                wait=wait,
                data_capture_s3_prefix=data_capture_s3_prefix,
                data_capture_sample_percent=data_capture_sample_percent,
            )
        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        _echo(
            f'Successfully created AWS Sagemaker deployment {name}', CLI_COLOR_SUCCESS,
        )
        _print_deployment_info(result.deployment, output)

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
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which '
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
    @click.option(
        '--timeout',
        help="The amount of time Sagemaker will wait before return response",
        type=click.INT,
        default=60,
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for apply action to complete or encounter an error.'
        'If set to no-wait, CLI will return immediately. The default value is wait',
    )
    @click.option(
        '--data-capture-s3-prefix',
        help="To enable data capture (input and output), "
        "provide a destination s3 prefix "
        "(of the format `s3://bucket-name/optional/prefix`)"
        " for the captured data. To disable data capture, leave this blank.",
        type=click.STRING,
        default=None,
    )
    @click.option(
        '--data-capture-sample-percent',
        help="When data capture is enabled, the sampling percentage. Default 100%. "
        "No effect without data-capture-s3-prefix.",
        type=click.IntRange(1, 100),
        default=100,
    )
    def update(
        name,
        namespace,
        bento,
        api_name,
        instance_type,
        instance_count,
        num_of_gunicorn_workers_per_instance,
        timeout,
        output,
        wait,
        data_capture_s3_prefix,
        data_capture_sample_percent,
    ):
        _echo(
            message='AWS Sagemaker deployment functionalities are being migrated to a '
            'separate tool and related CLI commands will be deprecated in BentoML '
            'itself, please use https://github.com/bentoml/aws-sagemaker-deploy '
            'going forward.',
            color='yellow',
        )
        yatai_client = get_default_yatai_client()
        if bento:
            bento_name, bento_version = bento.split(':')
        else:
            bento_name = None
            bento_version = None
        with Spinner('Updating Sagemaker deployment '):
            result = yatai_client.deployment.update_sagemaker_deployment(
                namespace=namespace,
                deployment_name=name,
                bento_name=bento_name,
                bento_version=bento_version,
                instance_count=instance_count,
                instance_type=instance_type,
                num_of_gunicorn_workers_per_instance=num_of_gunicorn_workers_per_instance,  # noqa E501
                timeout=timeout,
                api_name=api_name,
                wait=wait,
                data_capture_s3_prefix=data_capture_s3_prefix,
                data_capture_sample_percent=data_capture_sample_percent,
            )
        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        _echo(
            f'Successfully updated AWS Sagemaker deployment {name}', CLI_COLOR_SUCCESS,
        )
        _print_deployment_info(result.deployment, output)

    @aws_sagemaker.command(help='Delete AWS Sagemaker deployment')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which '
        'can be changed in BentoML configuration yatai_service/default_namespace',
    )
    @click.option(
        '--force',
        is_flag=True,
        help='force delete the deployment record in database and '
        'ignore errors when deleting cloud resources',
    )
    def delete(name, namespace, force):
        _echo(
            message='AWS Sagemaker deployment functionalities are being migrated to a '
            'separate tool and related CLI commands will be deprecated in BentoML '
            'itself, please use https://github.com/bentoml/aws-sagemaker-deploy '
            'going forward.',
            color='yellow',
        )
        yatai_client = get_default_yatai_client()
        get_deployment_result = yatai_client.deployment.get(namespace, name)
        if get_deployment_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_deployment_result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        result = yatai_client.deployment.delete(name, namespace, force)
        if result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        _echo(
            f'Successfully deleted AWS Sagemaker deployment "{name}"',
            CLI_COLOR_SUCCESS,
        )

    @aws_sagemaker.command(help='Get AWS Sagemaker deployment information')
    @click.argument('name', type=click.STRING)
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which '
        'can be changed in BentoML configuration yatai_service/default_namespace',
    )
    @click.option(
        '-o', '--output', type=click.Choice(['json', 'yaml', 'table']), default='json'
    )
    def get(name, namespace, output):
        _echo(
            message='AWS Sagemaker deployment functionalities are being migrated to a '
            'separate tool and related CLI commands will be deprecated in BentoML '
            'itself, please use https://github.com/bentoml/aws-sagemaker-deploy '
            'going forward.',
            color='yellow',
        )
        yatai_client = get_default_yatai_client()
        get_result = yatai_client.deployment.get(namespace, name)
        if get_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        describe_result = yatai_client.deployment.describe(namespace, name)
        if describe_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                describe_result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        get_result.deployment.state.CopyFrom(describe_result.state)
        _print_deployment_info(get_result.deployment, output)

    @aws_sagemaker.command(
        name='list', help='List AWS Sagemaker deployment information'
    )
    @click.option(
        '-n',
        '--namespace',
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which '
        'can be changed in BentoML configuration yatai_service/default_namespace',
        default=ALL_NAMESPACE_TAG,
    )
    @click.option(
        '--limit',
        type=click.INT,
        help='The maximum amount of AWS Sagemaker deployments to be listed at once',
    )
    @click.option(
        '-l',
        '--labels',
        type=click.STRING,
        help="Label query to filter Sagemaker deployments, supports '=', '!=', 'IN', "
        "'NotIn', 'Exists', and 'DoesNotExist'. (e.g. key1=value1, "
        "key2!=value2, key3 In (value3, value3a), key4 DoesNotExist)",
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
        '-o',
        '--output',
        type=click.Choice(['json', 'yaml', 'table', 'wide']),
        default='table',
    )
    def list_deployment(namespace, limit, labels, order_by, asc, output):
        _echo(
            message='AWS Sagemaker deployment functionalities are being migrated to a '
            'separate tool and related CLI commands will be deprecated in BentoML '
            'itself, please use https://github.com/bentoml/aws-sagemaker-deploy '
            'going forward.',
            color='yellow',
        )
        yatai_client = get_default_yatai_client()
        list_result = yatai_client.deployment.list_sagemaker_deployments(
            limit=limit,
            labels=labels,
            namespace=namespace,
            order_by=order_by,
            ascending_order=asc,
        )
        if list_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                list_result.status
            )
            raise CLIException(f'{error_code}:{error_message}')
        _print_deployments_info(list_result.deployments, output)

    return aws_sagemaker
