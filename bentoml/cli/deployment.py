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

from bentoml.deployment.serverless import ServerlessDeployment
from bentoml.deployment.sagemaker import SagemakerDeployment
from bentoml.cli.click_utils import _echo, CLI_COLOR_ERROR, CLI_COLOR_SUCCESS
from bentoml.utils.usage_stats import track_cli

SERVERLESS_PLATFORMS = ["aws-lambda", "aws-lambda-py2", "gcp-function"]

# pylint: disable=unused-variable


def add_deployment_commands(cli):

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
