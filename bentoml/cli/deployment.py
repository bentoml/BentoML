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

from google.protobuf.json_format import MessageToJson
from bentoml.deployment.serverless import ServerlessDeployment
from bentoml.deployment.sagemaker import SagemakerDeployment
from bentoml.cli.click_utils import (
    _echo,
    CLI_COLOR_ERROR,
    CLI_COLOR_SUCCESS,
    parse_bento_tag_callback,
)
from bentoml.yatai import get_yatai_service
from bentoml.proto.deployment_pb2 import (
    ApplyDeploymentRequest,
    DeleteDeploymentRequest,
    GetDeploymentRequest,
    DescribeDeploymentRequest,
    ListDeploymentsRequest,
    Deployment,
    DeploymentSpec,
    DeploymentOperator,
)
from bentoml.proto.status_pb2 import Status
from bentoml.utils import pb_to_yaml
from bentoml.utils.usage_stats import track_cli
from bentoml.exceptions import BentoMLDeploymentException, BentoMLException

SERVERLESS_PLATFORMS = ["aws-lambda", "aws-lambda-py2", "gcp-function"]

# pylint: disable=unused-variable

logger = logging.getLogger(__name__)


def add_legacy_deployment_commands(cli):

    # Example usage: bentoml deploy /ARCHIVE_PATH --platform=aws-lambda
    @cli.command(
        help="Deploy BentoML archive as REST endpoint to cloud services",
        short_help="Deploy Bento archive",
    )
    @click.argument("archive-path", type=click.STRING)
    @click.option(
        "--platform",
        type=click.Choice(
            [
                "aws-lambda",
                "aws-lambda-py2",
                "gcp-function",
                "aws-sagemaker",
                "azure-ml",
                "algorithmia",
            ]
        ),
        required=True,
        help="Target platform that Bento archive is going to deployed to",
    )
    @click.option(
        "--region",
        type=click.STRING,
        help="Target region inside the cloud provider that will be deployed to",
    )
    @click.option("--stage", type=click.STRING)
    @click.option(
        "--api-name", type=click.STRING, help="The name of API will be deployed"
    )
    @click.option(
        "--instance-type",
        type=click.STRING,
        help="SageMaker deployment ONLY. The instance type to use for deployment",
    )
    @click.option(
        "--instance-count",
        type=click.INT,
        help="Sagemaker deployment ONLY. Number of instances to use for deployment",
    )
    def deploy(
        archive_path, platform, region, stage, api_name, instance_type, instance_count
    ):
        track_cli("deploy", platform)
        if platform in SERVERLESS_PLATFORMS:
            deployment = ServerlessDeployment(archive_path, platform, region, stage)
        elif platform == "aws-sagemaker":
            deployment = SagemakerDeployment(
                archive_path, api_name, region, instance_count, instance_type
            )
        else:
            _echo(
                "Deploying with --platform=%s is not supported in current version of "
                "BentoML" % platform,
                CLI_COLOR_ERROR,
            )
            return

        try:
            output_path = deployment.deploy()

            _echo(
                "Successfully deployed to {platform}!".format(platform=platform),
                CLI_COLOR_SUCCESS,
            )
            _echo("Deployment archive is saved at: %s" % output_path)
        except Exception as e:  # pylint:disable=broad-except
            _echo(
                "Encounter error when deploying to {platform}\nError: {error_message}".format(
                    platform=platform, error_message=str(e)
                ),
                CLI_COLOR_ERROR,
            )

    # Example usage: bentoml delete-deployment ARCHIVE_PATH --platform=aws-lambda
    @cli.command(
        help="Delete active BentoML deployment from cloud services",
        short_help="Delete active BentoML deployment",
    )
    @click.argument("archive-path", type=click.STRING)
    @click.option(
        "--platform",
        type=click.Choice(
            [
                "aws-lambda",
                "aws-lambda-py2",
                "gcp-function",
                "aws-sagemaker",
                "azure-ml",
                "algorithmia",
            ]
        ),
        required=True,
        help="The platform bento archive is deployed to",
    )
    @click.option(
        "--region",
        type=click.STRING,
        required=True,
        help="The region deployment belongs to",
    )
    @click.option(
        "--api-name",
        type=click.STRING,
        help="Name of the API function that is deployed",
    )
    @click.option("--stage", type=click.STRING)
    def delete_deployment(archive_path, platform, region, stage, api_name):
        track_cli("delete-deploy", platform)
        if platform in SERVERLESS_PLATFORMS:
            deployment = ServerlessDeployment(archive_path, platform, region, stage)
        elif platform == "aws-sagemaker":
            deployment = SagemakerDeployment(archive_path, api_name, region)
        else:
            _echo(
                "Remove deployment with --platform=%s is not supported in current "
                "version of BentoML" % platform,
                CLI_COLOR_ERROR,
            )
            return

        if deployment.delete():
            _echo(
                "Successfully delete {platform} deployment".format(platform=platform),
                CLI_COLOR_SUCCESS,
            )
        else:
            _echo(
                "Delete {platform} deployment unsuccessful".format(platform=platform),
                CLI_COLOR_ERROR,
            )

    # Example usage: bentoml check-deployment-status ARCHIVE_PATH --platform=aws-lambda
    @cli.command(
        help="Check deployment status of BentoML archive",
        short_help="check deployment status",
    )
    @click.argument("archive-path", type=click.STRING)
    @click.option(
        "--platform",
        type=click.Choice(
            [
                "aws-lambda",
                "aws-lambda-py2",
                "gcp-function",
                "aws-sagemaker",
                "azure-ml",
                "algorithmia",
            ]
        ),
        required=True,
        help="Target platform that Bento archive will be deployed to as a REST api \
                service",
    )
    @click.option(
        "--region",
        type=click.STRING,
        required=True,
        help="Deployment's region name inside cloud provider.",
    )
    @click.option("--stage", type=click.STRING)
    @click.option(
        "--api-name",
        type=click.STRING,
        help="The name of API that is deployed as a service.",
    )
    def check_deployment_status(archive_path, platform, region, stage, api_name):
        track_cli("check-deployment-status", platform)
        if platform in SERVERLESS_PLATFORMS:
            deployment = ServerlessDeployment(archive_path, platform, region, stage)
        elif platform == "aws-sagemaker":
            deployment = SagemakerDeployment(archive_path, api_name, region)
        else:
            _echo(
                "check deployment status with --platform=%s is not supported in the "
                "current version of BentoML" % platform,
                CLI_COLOR_ERROR,
            )
            return

        deployment.check_status()

    return cli


def parse_key_value_pairs(key_value_pairs_str):
    result = {}
    if key_value_pairs_str:
        for key_value_pair in key_value_pairs_str.split(','):
            key, value = key_value_pair.split('=')
            key = key.strip()
            value = value.strip()
            if key in result:
                logger.warning('duplicated key "%s" found string map parameter', key)
            result[key] = value
    return result


def get_deployment_operator_type(platform):
    return DeploymentOperator.Value(platform.upper())


def display_deployment_info(deployment, output):
    if output == 'yaml':
        result = pb_to_yaml(deployment)
    else:
        result = MessageToJson(deployment)
    _echo(result)


def parse_bento_tag(tag):
    items = tag.split(':')

    if len(items) > 2:
        raise BentoMLException("More than one ':' appeared in tag '%s'" % tag)
    elif len(items) == 1:
        return tag, 'latest'
    else:
        if not items[0]:
            raise BentoMLException("':' can't be the leading character")
        if not items[1]:
            raise BentoMLException("Please include value for the key %s" % items[0])
        return items[0], items[1]


def get_deployment_sub_command(cli):
    @click.group()
    def deploy():
        pass

    @deploy.command(
        short_help="Create or update a model serving deployment",
        context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
    )
    @click.argument("--deployment-name", type=click.STRING, required=True)
    @click.option(
        '--bento',
        type=click.STRING,
        required=True,
        callback=parse_bento_tag_callback,
        help="Deployed bento archive, in format of name:version.  For example, iris_classifier:v1.2.0",
    )
    @click.option(
        "--platform",
        type=click.Choice(
            ["aws_lambda", "gcp_function", "aws_sagemaker", "kubernetes", "custom"]
        ),
        required=True,
        help="Target platform that Bento archive is going to deployed to",
    )
    @click.option("--namespace", type=click.STRING, help="Deployment's namespace")
    @click.option(
        "--labels",
        type=click.STRING,
        help="Key:value pairs that attached to deployment.",
    )
    @click.option("--annotations", type=click.STRING)
    @click.option(
        '--region',
        help="Name of the deployed region. For platforms: AWS_Lambda, AWS_SageMaker, GCP_Function",
    )
    @click.option(
        '--stage', help="Stage is to identify. For platform:  AWS_Lambda, GCP_Function"
    )
    @click.option(
        '--instance-type',
        help="Type of instance will be used for inference. For platform: AWS_SageMaker",
    )
    @click.option(
        '--instance-count',
        help="Number of instance will be used. For platform: AWS_SageMaker",
    )
    @click.option(
        '--api-name',
        help="User defined API function will be used for inference. For platform: AWS_SageMaker",
    )
    @click.option(
        '--kube-namespace',
        help="Namespace for kubernetes deployment. For platform: Kubernetes",
    )
    @click.option('--replicas', help="Number of replicas. For platform: Kubernetes")
    @click.option('--service-name', help="Name for service. For platform: Kubernetes")
    @click.option('--service-type', help="Service Type. For platform: Kubernetes")
    @click.option('--output', type=click.Choice(['json', 'yaml']), default='json')
    def apply(
        bento,
        deployment_name,
        platform,
        output,
        namespace=None,
        labels=None,
        annotations=None,
        region=None,
        stage=None,
        instance_type=None,
        instance_count=None,
        api_name=None,
        kube_namespace=None,
        replicas=None,
        service_name=None,
        service_type=None,
    ):
        track_cli('deploy-apply', platform)
        bento_name, bento_verison = bento.split(':')
        spec = DeploymentSpec(
            bento_name=bento_name,
            bento_verison=bento_verison,
            operator=get_deployment_operator_type(platform),
        )
        if platform == 'aws_sagemaker':
            spec.sagemaker_operator_config = DeploymentSpec.SageMakerOperatorConfig(
                region=region,
                instance_count=instance_count,
                instance_type=instance_type,
                api_name=api_name,
            )
        elif platform == 'aws_lambda':
            spec.aws_lambda_operator_config = DeploymentSpec.AwsLambdaOperatorConfig(
                region=region, stage=stage
            )
        elif platform == 'gcp_function':
            spec.gcp_function_operator_config = DeploymentSpec.GcpFunctionOperatorConfig(
                region=region, stage=stage
            )
        elif platform == 'kubernetes':
            spec.kubernetes_operator_config = DeploymentSpec.KubernetesOperatorConfig(
                kube_namespace=kube_namespace,
                replicas=replicas,
                service_name=service_name,
                service_type=service_type,
            )
        else:
            raise BentoMLDeploymentException(
                "Custom deployment configuration isn't supported in the current version"
            )

        result = get_yatai_service().ApplyDeployment(
            ApplyDeploymentRequest(
                deployment=Deployment(
                    namespace=namespace,
                    name=deployment_name,
                    annotations=parse_key_value_pairs(annotations),
                    labels=parse_key_value_pairs(labels),
                    spec=spec,
                )
            )
        )
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to apply deployment {name}. code: {error_code}, message: {error_message}'.format(
                    name=deployment_name,
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            _echo(
                'Successfully apply deployment {}'.format(deployment_name),
                CLI_COLOR_SUCCESS,
            )
            display_deployment_info(result.deployment, output)

    @deploy.command()
    @click.option("--name", type=click.STRING, help="Deployment name", required=True)
    def delete(name):
        track_cli('deploy-delete')
        result = get_yatai_service().DeleteDeployment(
            DeleteDeploymentRequest(deployment_name=name)
        )
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to delete deployment {name}. code: {error_code}, message: {error_message}'.format(
                    name=name,
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            _echo('Successfully delete deployment {}'.format(name), CLI_COLOR_SUCCESS)

    @deploy.command()
    @click.option("--name", type=click.STRING, help="Deployment name", required=True)
    @click.option('--output', type=click.Choice(['json', 'yaml']), default='json')
    def get(name, output):
        track_cli('deploy-get')
        result = get_yatai_service().GetDeployment(
            GetDeploymentRequest(deployment_name=name)
        )
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to get deployment {name}. code: {error_code}, message: {error_message}'.format(
                    name=name,
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            display_deployment_info(result.deployment, output)

    @deploy.command()
    @click.option("--name", type=click.STRING, help="Deployment name", required=True)
    @click.option('--output', type=click.Choice(['json', 'yaml']), default='json')
    def describe(name, output=None):
        track_cli('deploy-describe')
        result = get_yatai_service().DescribeDeployment(
            DescribeDeploymentRequest(deployment_name=name)
        )
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to describe deployment {name}. code: {error_code}, message: {error_message}'.format(
                    name=name,
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            display_deployment_info(result.deployment, output)

    @deploy.command()
    @click.option(
        "--limit", type=click.INT, help="Limit how many deployments will be retrieved"
    )
    @click.option(
        "--filter", type=click.STRING, help="Filter retrieved deployments with keywords"
    )
    @click.option(
        "--labels", type=click.STRING, help="List deployments with the giving labels"
    )
    @click.option('--output', type=click.Choice(['json', 'yaml']), default='json')
    def list(output, limit=None, filter=None, labels=None):
        track_cli('deploy-list')
        result = get_yatai_service().ListDeployments(
            ListDeploymentsRequest(
                limit=limit, filter=filter, labels=parse_key_value_pairs(labels)
            )
        )
        if result.status.status_code != Status.OK:
            _echo(
                'Failed to list deployments. code: {error_code}, message: {error_message}'.format(
                    error_code=Status.Code.Name(result.status.status_code),
                    error_message=result.status.error_message,
                ),
                CLI_COLOR_ERROR,
            )
        else:
            for deployment_pb in result.deployments:
                display_deployment_info(deployment_pb, output)

    return deploy
