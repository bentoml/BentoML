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
import grpc

from bentoml.config import config
from bentoml.deployment.serverless import ServerlessDeployment
from bentoml.deployment.sagemaker import SagemakerDeployment
from bentoml.cli.click_utils import _echo, CLI_COLOR_ERROR, CLI_COLOR_SUCCESS
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
from bentoml.utils.usage_stats import track_cli

SERVERLESS_PLATFORMS = ["aws-lambda", "aws-lambda-py2", "gcp-function"]

# pylint: disable=unused-variable


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


def parse_key_value_string(key_value_string):
    if key_value_string:
        result = {}
        for item in key_value_string.split(','):
            if item is not None:
                splits = item.split('=')
                result[splits[0]] = splits[1]
        return result
    else:
        return None


def get_deployment_operator_type(platform):
    return DeploymentOperator.Value(platform.upper())


def display_response_status_error(status):
    _echo('Error code: {}'.format(status.status_code), CLI_COLOR_ERROR)
    _echo('Error message: {}'.format(status.error_message), CLI_COLOR_ERROR)


def display_deployment_info(deployment):
    _echo('deployment info')

def get_deployment_sub_command(cli):
    @click.group()
    def deployment():
        pass

    @deployment.command(
        help="Apply deployment configuration to target deployment platform",
        short_help="Apply deployment configuration",
    )
    @click.argument("archive-path", type=click.STRING)
    @click.option("--namespace", type=click.STRING, help="Deployment's namespace")
    @click.option("--name", type=click.STRING, help="Deployment's name")
    @click.option("--labels", type=click.STRING)
    @click.option("--annotations", type=click.STRING)
    @click.option(
        "--platform",
        type=click.Choice(
            ["aws_lambda", "gcp_function", "aws_sagemaker", "kubernetes", "custom"]
        ),
        required=True,
        help="Target platform that Bento archive is going to deployed to",
    )
    @click.option(
        "--region",
        type=click.STRING,
        help="Target region inside the cloud provider that will be deployed to",
    )
    @click.pass_context
    def apply(
        ctx,
        archive_path,
        platform,
        region=None,
        name=None,
        namespace=None,
        labels=None,
        annotations=None,
    ):
        bento = 'store.get(archive_path)'
        operator_config = {}
        spec = {
            "bento_name": bento.name,
            "bento_version": bento.version,
            "operator": get_deployment_operator_type(platform),
            "deployment_operator_config": operator_config,
        }
        result = get_yatai_service().ApplyDeployment(
            request={
                'deployment': {
                    "namespace": namespace,
                    "name": name,
                    "annotations": parse_key_value_string(annotations),
                    "labels": parse_key_value_string(labels),
                    "spec": spec,
                }
            }
        )
        if result.status.status_code != Status.OK:
            _echo('Apply deployment {} failed'.format(name), CLI_COLOR_ERROR)
            display_response_status_error(result.status)
        else:
            _echo('Successful apply deployment {}'.format(name), CLI_COLOR_SUCCESS)
            display_deployment_info(result.deployment)


    @deployment.command()
    @click.option("--name", type=click.STRING, help="Deployment's name", required=True)
    def delete(name):
        result = get_yatai_service().DeleteDeployment(request={'deployment_name': name})
        if result.status.status_code != Status.OK:
            _echo('Delete deployment {} failed'.format(name), CLI_COLOR_ERROR)
            display_response_status_error(result.status)
        else:
            _echo('Successful delete deployment {}'.format(name), CLI_COLOR_SUCCESS)

    @deployment.command()
    @click.option("--name", type=click.STRING, help="Deployment's name", required=True)
    def get(name):
        result = get_yatai_service().GetDeployment(request={'deployment_name': name})
        if result.status.status_code != Status.OK:
            _echo('Get deployment {} failed'.format(name), CLI_COLOR_ERROR)
            display_response_status_error(result.status)
        else:
            display_deployment_info(result.deployment)

    @deployment.command()
    @click.option("--name", type=click.STRING, help="Deployment's name", required=True)
    def describe(name):
        result = get_yatai_service().DescribeDeployment(
            request={'deployment_name': name}
        )
        if result.status.status_code != Status.OK:
            _echo('Describe deployment {} failed'.format(name), CLI_COLOR_ERROR)
            display_response_status_error(result.status)
        else:
            display_deployment_info(result.deployment)

    @deployment.command()
    @click.option("--limit", type=click.INT, help="")
    @click.option("--offset", type=click.INT, help="")
    @click.option("--filter", type=click.STRING, help="")
    @click.option("--labels", type=click.STRING, help="")
    def list(limit=None, offset=None, filter=None, labels=None):
        result = get_yatai_service().ListDeployments(
            request={
                'limit': limit,
                'offset': offset,
                'filter': filter,
                'labels': parse_key_value_string(labels),
            }
        )
        if result.status.status_code != Status.OK:
            _echo('List deployments failed', CLI_COLOR_ERROR)
            display_response_status_error(result.status)
        else:
            for deployment in result.deployments:
                display_deployment_info(deployment)

    return deployment
