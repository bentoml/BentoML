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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import logging
import time

from google.protobuf.json_format import MessageToJson
from bentoml.cli.click_utils import (
    _echo,
    CLI_COLOR_ERROR,
    CLI_COLOR_SUCCESS,
    parse_bento_tag_callback,
    parse_yaml_file_callback,
)
from bentoml.cli.deployment_utils import deployment_yaml_to_pb
from bentoml.yatai import get_yatai_service
from bentoml.proto.deployment_pb2 import (
    ApplyDeploymentRequest,
    DeleteDeploymentRequest,
    GetDeploymentRequest,
    DescribeDeploymentRequest,
    ListDeploymentsRequest,
    Deployment,
    DeploymentSpec,
    DeploymentState,
)
from bentoml.proto.status_pb2 import Status
from bentoml.utils import pb_to_yaml
from bentoml.utils.usage_stats import track_cli
from bentoml.exceptions import BentoMLDeploymentException, BentoMLException
from bentoml.deployment.store import ALL_NAMESPACE_TAG
from bentoml import config

# pylint: disable=unused-variable

logger = logging.getLogger(__name__)


def parse_key_value_pairs(key_value_pairs_str):
    result = {}
    if key_value_pairs_str:
        for key_value_pair in key_value_pairs_str.split(','):
            key, value = key_value_pair.split('=')
            key = key.strip()
            value = value.strip()
            if key in result:
                logger.warning("duplicated key '%s' found string map parameter", key)
            result[key] = value
    return result


def display_deployment_info(deployment, output):
    if output == 'yaml':
        result = pb_to_yaml(deployment)
    else:
        result = MessageToJson(deployment)
    _echo(result)


def get_state_after_await_action_complete(
    yatai_service, name, namespace, message, timeout_limit=600, wait_time=5
):
    start_time = time.time()
    while (time.time() - start_time) < timeout_limit:
        result = yatai_service.DescribeDeployment(
            DescribeDeploymentRequest(deployment_name=name, namespace=namespace)
        )
        if result.state.state is DeploymentState.PENDING:
            time.sleep(wait_time)
            _echo(message)
            continue
        else:
            break
    return result


def get_deployment_sub_command():
    @click.group(help='Deployment commands. Shorthand is `deploy`')
    def deployment():
        pass

    @deployment.command(
        short_help='Create a model serving deployment',
        context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
    )
    @click.argument("name", type=click.STRING, required=True)
    @click.option(
        '--bento',
        type=click.STRING,
        required=True,
        callback=parse_bento_tag_callback,
        help='Deployed bento archive, in format of name:version. For example, '
        'iris_classifier:v1.2.0',
    )
    @click.option(
        '--platform',
        type=click.Choice(
            ['aws-lambda', 'gcp-function', 'aws-sagemaker', 'kubernetes', 'custom'],
            case_sensitive=False,
        ),
        required=True,
        help='Target platform that Bento archive is going to deployed to',
    )
    @click.option('--namespace', type=click.STRING, help='Deployment namespace')
    @click.option(
        '--labels',
        type=click.STRING,
        help='Key:value pairs that attached to deployment.',
    )
    @click.option('--annotations', type=click.STRING)
    @click.option(
        '--region',
        help='Name of the deployed region. For platforms: AWS Lambda, AWS SageMaker, '
        'GCP Function',
    )
    @click.option(
        '--instance-type',
        help='Type of instance will be used for inference. For platform: AWS SageMaker',
    )
    @click.option(
        '--instance-count',
        help='Number of instance will be used. For platform: AWS SageMaker',
        type=click.INT,
    )
    @click.option(
        '--api-name',
        help='User defined API function will be used for inference. For platform: '
        'AWS SageMaker',
    )
    @click.option(
        '--kube-namespace',
        help='Namespace for kubernetes deployment. For platform: Kubernetes',
    )
    @click.option('--replicas', help='Number of replicas. For platform: Kubernetes')
    @click.option('--service-name', help='Name for service. For platform: Kubernetes')
    @click.option('--service-type', help='Service Type. For platform: Kubernetes')
    @click.option('--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for apply action to complete or encounter an error.'
        'If set to no-wait, CLI will return immediately. The default value is wait',
    )
    def create(
        name,
        bento,
        platform,
        output,
        namespace,
        labels,
        annotations,
        region,
        instance_type,
        instance_count,
        api_name,
        kube_namespace,
        replicas,
        service_name,
        service_type,
        wait,
    ):
        # converting platform parameter to DeploymentOperator name in proto
        # e.g. 'aws-lambda' to 'AWS_LAMBDA'
        platform = platform.replace('-', '_').upper()
        operator = DeploymentSpec.DeploymentOperator.Value(platform)

        track_cli('deploy-create', platform)

        yatai_service = get_yatai_service()

        # Make sure there is no active deployment with the same deployment name
        get_deployment = yatai_service.GetDeployment(
            GetDeploymentRequest(deployment_name=name, namespace=namespace)
        )
        if get_deployment.status.status_code != Status.NOT_FOUND:
            raise BentoMLDeploymentException(
                'Deployment {name} already existed, please use update or apply command'
                ' instead'.format(name=name)
            )

        if operator == DeploymentSpec.AWS_SAGEMAKER:
            if not api_name:
                raise click.BadParameter(
                    'api-name is required for Sagemaker deployment'
                )

            sagemaker_operator_config = DeploymentSpec.SageMakerOperatorConfig(
                region=region or config().get('aws', 'default_region'),
                instance_count=instance_count
                or config().getint('sagemaker', 'instance_count'),
                instance_type=instance_type
                or config().get('sagemaker', 'instance_type'),
                api_name=api_name,
            )
            spec = DeploymentSpec(sagemaker_operator_config=sagemaker_operator_config)
        elif operator == DeploymentSpec.AWS_LAMBDA:
            aws_lambda_operator_config = DeploymentSpec.AwsLambdaOperatorConfig(
                region=region or config().get('aws', 'default_region')
            )
            spec = DeploymentSpec(aws_lambda_operator_config=aws_lambda_operator_config)
        elif operator == DeploymentSpec.GCP_FUNCTION:
            gcp_function_operator_config = DeploymentSpec.GcpFunctionOperatorConfig(
                region=region or config().get('google-cloud', 'default_region')
            )
            spec = DeploymentSpec(
                gcp_function_operator_config=gcp_function_operator_config
            )
        elif operator == DeploymentSpec.KUBERNETES:
            kubernetes_operator_config = DeploymentSpec.KubernetesOperatorConfig(
                kube_namespace=kube_namespace,
                replicas=replicas,
                service_name=service_name,
                service_type=service_type,
            )
            spec = DeploymentSpec(kubernetes_operator_config=kubernetes_operator_config)
        else:
            raise BentoMLDeploymentException(
                'Custom deployment is not supported in the current version of BentoML'
            )

        bento_name, bento_version = bento.split(':')
        spec.bento_name = bento_name
        spec.bento_version = bento_version
        spec.operator = operator

        result = yatai_service.ApplyDeployment(
            ApplyDeploymentRequest(
                deployment=Deployment(
                    namespace=namespace,
                    name=name,
                    annotations=parse_key_value_pairs(annotations),
                    labels=parse_key_value_pairs(labels),
                    spec=spec,
                )
            )
        )
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to create deployment {name}. code: {error_code}, message: '
                '{error_message}'.format(
                    name=name,
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            if wait:
                result_state = get_state_after_await_action_complete(
                    yatai_service=yatai_service,
                    name=name,
                    namespace=namespace,
                    message='Creating deployment...',
                )
                result.deployment.state.CopyFrom(result_state.state)

            _echo('Finished create deployment {}'.format(name), CLI_COLOR_SUCCESS)
            display_deployment_info(result.deployment, output)

    @deployment.command(help='Apply model service deployment from yaml file')
    @click.option(
        '-f',
        '--file',
        'deployment_yaml',
        type=click.File('r'),
        callback=parse_yaml_file_callback,
    )
    @click.option('-o', '--output', type=click.Choice(['json', 'yaml']), default='json')
    @click.option(
        '--wait/--no-wait',
        default=True,
        help='Wait for apply action to complete or encounter an error.'
        'If set to no-wait, CLI will return immediately. The default value is wait',
    )
    def apply(deployment_yaml, output, wait):
        track_cli('deploy-apply', deployment_yaml.get('spec', {}).get('operator'))
        try:
            deployment_pb = deployment_yaml_to_pb(deployment_yaml)
            yatai_service = get_yatai_service()
            result = yatai_service.ApplyDeployment(
                ApplyDeploymentRequest(deployment=deployment_pb)
            )
            if result.status.status_code != Status.OK:
                _echo(
                    'Failed to apply deployment {name}. code: {error_code}, message: '
                    '{error_message}'.format(
                        name=deployment_pb.name,
                        error_code=Status.Code.Name(result.status.status_code),
                        error_message=result.status.error_message,
                    ),
                    CLI_COLOR_ERROR,
                )
            else:
                if wait:
                    result_state = get_state_after_await_action_complete(
                        yatai_service=yatai_service,
                        name=deployment_pb.name,
                        namespace=deployment_pb.namespace,
                        message='Applying deployment...',
                    )
                    result.deployment.state.CopyFrom(result_state.state)

                _echo(
                    'Finished apply deployment {}'.format(deployment_pb.name),
                    CLI_COLOR_SUCCESS,
                )
                display_deployment_info(result.deployment, output)
        except BentoMLException as e:
            _echo(
                'Failed to apply deployment {name}. Error message: {message}'.format(
                    name=deployment_pb.name, message=e
                )
            )

    @deployment.command(help='Delete deployment')
    @click.argument("name", type=click.STRING, required=True)
    @click.option('--namespace', type=click.STRING, help='Deployment namespace')
    def delete(name, namespace):
        track_cli('deploy-delete')

        result = get_yatai_service().DeleteDeployment(
            DeleteDeploymentRequest(deployment_name=name, namespace=namespace)
        )
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to delete deployment {name}. code: {error_code}, message: '
                '{error_message}'.format(
                    name=name,
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            _echo('Successfully delete deployment {}'.format(name), CLI_COLOR_SUCCESS)

    @deployment.command(help='Get deployment spec')
    @click.argument("name", type=click.STRING, required=True)
    @click.option('--namespace', type=click.STRING, help='Deployment namespace')
    @click.option('--output', type=click.Choice(['json', 'yaml']), default='json')
    def get(name, output, namespace):
        track_cli('deploy-get')

        result = get_yatai_service().GetDeployment(
            GetDeploymentRequest(deployment_name=name, namespace=namespace)
        )
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to get deployment {name}. code: {error_code}, message: '
                '{error_message}'.format(
                    name=name,
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            display_deployment_info(result.deployment, output)

    @deployment.command(help='Get deployment state')
    @click.argument("name", type=click.STRING, required=True)
    @click.option('--namespace', type=click.STRING, help='Deployment namespace')
    @click.option('--output', type=click.Choice(['json', 'yaml']), default='json')
    def describe(name, output, namespace):
        track_cli('deploy-describe')
        yatai_service = get_yatai_service()

        result = yatai_service.DescribeDeployment(
            DescribeDeploymentRequest(deployment_name=name, namespace=namespace)
        )
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to describe deployment {name}. code: {error_code}, message: '
                '{error_message}'.format(
                    name=name,
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            get_response = yatai_service.GetDeployment(
                GetDeploymentRequest(deployment_name=name, namespace=namespace)
            )
            deployment_pb = get_response.deployment
            deployment_pb.state.CopyFrom(result.state)
            display_deployment_info(deployment_pb, output)

    @deployment.command(name="list", help='List deployments')
    @click.option('--namespace', type=click.STRING)
    @click.option('--all-namespace', type=click.BOOL, default=False)
    @click.option(
        '--limit', type=click.INT, help='Limit how many deployments will be retrieved'
    )
    @click.option(
        '--filters',
        type=click.STRING,
        help='Filter retrieved deployments with keywords',
    )
    @click.option(
        '--labels', type=click.STRING, help='List deployments with the giving labels'
    )
    @click.option('--output', type=click.Choice(['json', 'yaml']), default='json')
    def list_deployments(output, limit, filters, labels, namespace, all_namespace):
        track_cli('deploy-list')

        if all_namespace:
            if namespace is not None:
                logger.warning(
                    'Ignoring `namespace=%s` due to the --all-namespace flag presented',
                    namespace,
                )
            namespace = ALL_NAMESPACE_TAG

        result = get_yatai_service().ListDeployments(
            ListDeploymentsRequest(
                limit=limit,
                filter=filters,
                labels=parse_key_value_pairs(labels),
                namespace=namespace,
            )
        )
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to list deployments. code: {error_code}, message: '
                '{error_message}'.format(
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            for deployment_pb in result.deployments:
                display_deployment_info(deployment_pb, output)

    return deployment
