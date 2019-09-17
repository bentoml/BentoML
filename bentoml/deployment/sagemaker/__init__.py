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

import os
import shutil
import base64
import logging
import re
import json
from six.moves.urllib.parse import urlparse

import boto3
import docker

from bentoml import config
from bentoml.deployment.utils import process_docker_api_line
from bentoml.yatai.status import Status
from bentoml.utils.tempdir import TempDirectory
from bentoml.exceptions import BentoMLDeploymentException
from bentoml.deployment.sagemaker.templates import (
    DEFAULT_NGINX_CONFIG,
    DEFAULT_WSGI_PY,
    DEFAULT_SERVE_SCRIPT,
)
from bentoml.deployment.operator import DeploymentOperatorBase
from bentoml.proto.deployment_pb2 import (
    Deployment,
    ApplyDeploymentResponse,
    DeleteDeploymentResponse,
    DescribeDeploymentResponse,
    DeploymentState,
)

logger = logging.getLogger(__name__)


DEFAULT_REGION = "us-west-2"
DEFAULT_INSTANCE_TYPE = "ml.m4.xlarge"
DEFAULT_INSTANCE_COUNT = 1


def strip_scheme(url):
    """ Stripe url's schema
    e.g.   http://some.url/path -> some.url/path
    :param url: String
    :return: String
    """
    parsed = urlparse(url)
    scheme = "%s://" % parsed.scheme
    return parsed.geturl().replace(scheme, "", 1)


def generate_aws_compatible_string(item):
    pattern = re.compile("[^a-zA-Z0-9-]|_")
    return re.sub(pattern, "-", item)


def create_sagemaker_model_name(bento_name, bento_version):
    return generate_aws_compatible_string(
        "bentoml-{name}-{version}".format(name=bento_name, version=bento_version)
    )


def create_sagemaker_endpoint_config_name(bento_name, bento_version):
    return generate_aws_compatible_string(
        '{name}-{version}-configuration'.format(name=bento_name, version=bento_version)
    )


def get_arn_role_from_current_user():
    sts_client = boto3.client("sts")
    identity = sts_client.get_caller_identity()
    sts_arn = identity["Arn"]
    sts_arn_list = sts_arn.split(":")
    type_role = sts_arn_list[-1].split("/")
    iam_client = boto3.client("iam")
    if type_role[0] == "user":
        role_list = iam_client.list_roles()
        arn = None
        for role in role_list["Roles"]:
            policy_document = role["AssumeRolePolicyDocument"]
            statement = policy_document["Statement"][0]
            if (
                statement["Effect"] == "Allow"
                and statement["Principal"]["Service"] == "sagemaker.amazonaws.com"
            ):
                arn = role["Arn"]
        if arn is None:
            raise ValueError(
                "Can't find proper Arn role for Sagemaker, please create one and try "
                "again"
            )
        return arn
    elif type_role[0] == "role":
        role_response = iam_client.get_role(RoleName=type_role[1])
        return role_response["Role"]["Arn"]


def create_push_image_to_ecr(bento_name, bento_version, snapshot_path):
    """Create BentoService sagemaker image and push to AWS ECR

    Example: https://github.com/awslabs/amazon-sagemaker-examples/blob/\
        master/advanced_functionality/scikit_bring_your_own/container/build_and_push.sh
    1. get aws account info and login ecr
    2. create ecr repository, if not exist
    3. build tag and push docker image

    Args:
        bento_service(BentoService)
        snapshot_path(Path)

    Returns:
        str: AWS ECR Tag
    """
    ecr_client = boto3.client("ecr")
    token = ecr_client.get_authorization_token()
    logger.info("Getting docker login info from AWS")
    username, password = (
        base64.b64decode(token["authorizationData"][0]["authorizationToken"])
        .decode("utf-8")
        .split(":")
    )
    registry_url = token["authorizationData"][0]["proxyEndpoint"]
    auth_config_payload = {"username": username, "password": password}

    docker_api = docker.APIClient()

    image_name = bento_name.lower() + "-sagemaker"
    ecr_tag = strip_scheme(
        "{registry_url}/{image_name}:{version}".format(
            registry_url=registry_url, image_name=image_name, version=bento_version
        )
    )

    logger.info("Building docker image: %s", image_name)
    for line in docker_api.build(
        path=snapshot_path, dockerfile="Dockerfile-sagemaker", tag=image_name
    ):
        process_docker_api_line(line)

    try:
        ecr_client.describe_repositories(repositoryNames=[image_name])["repositories"]
    except ecr_client.exceptions.RepositoryNotFoundException:
        ecr_client.create_repository(repositoryName=image_name)

    if docker_api.tag(image_name, ecr_tag) is False:
        raise RuntimeError("Tag appeared to fail: " + ecr_tag)
    logger.info("Pushing image to AWS ECR at %s", ecr_tag)
    for line in docker_api.push(ecr_tag, stream=True, auth_config=auth_config_payload):
        process_docker_api_line(line)
    logger.info("Finished pushing image: %s", ecr_tag)
    return ecr_tag


class TemporarySageMakerContent(object):
    def __init__(self, archive_path, bento_name, bento_version, _cleanup=True):
        self.archive_path = archive_path
        self.bento_name = bento_name
        self.bento_version = bento_version
        self.temp_directory = TempDirectory()
        self._cleanup = _cleanup
        self.path = None

    def __enter__(self):
        self.generate()
        return self.path

    def generate(self):
        self.temp_directory.create()
        tempdir = self.temp_directory.path
        saved_path = os.path.join(tempdir, self.bento_name, self.bento_version)
        shutil.copytree(self.archive_path, saved_path)

        with open(os.path.join(saved_path, "nginx.conf"), "w") as f:
            f.write(DEFAULT_NGINX_CONFIG)
        with open(os.path.join(saved_path, "wsgi.py"), "w") as f:
            f.write(DEFAULT_WSGI_PY)
        with open(os.path.join(saved_path, "serve"), "w") as f:
            f.write(DEFAULT_SERVE_SCRIPT)

        # permission 755 is required for entry script 'serve'
        permission = "755"
        octal_permission = int(permission, 8)
        os.chmod(os.path.join(saved_path, "serve"), octal_permission)
        self.path = saved_path

    def cleanup(self):
        self.temp_directory.cleanup()
        self.path = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cleanup:
            self.cleanup()


# Sagemaker response status: 'OutOfService'|'Creating'|'Updating'|
#                            'SystemUpdating'|'RollingBack'|'InService'|
#                            'Deleting'|'Failed'
ENDPOINT_STATUS_TO_STATE = {
    "InService": DeploymentState.RUNNING,
    "Deleting": DeploymentState.INACTIVATED,
    "Creating": DeploymentState.PENDING,
    "Updating": DeploymentState.PENDING,
    "RollingBack": DeploymentState.PENDING,
    "SystemUpdating": DeploymentState.PENDING,
    "OutOfService": DeploymentState.INACTIVATED,
    "Failed": DeploymentState.ERROR,
}


class SageMakerDeploymentOperator(DeploymentOperatorBase):
    def apply(self, deployment_pb, repo, prev_deployment=None):
        deployment_spec = deployment_pb.spec
        sagemaker_config = deployment_spec.sagemaker_operator_config
        if sagemaker_config is None:
            raise BentoMLDeploymentException('Sagemaker configuration is missing.')

        archive_path = repo.get(
            deployment_spec.bento_name, deployment_spec.bento_version
        )

        # config = load_bentoml_config(bento_path)...

        sagemaker_client = boto3.client('sagemaker', sagemaker_config.region)

        with TemporarySageMakerContent(
            archive_path, deployment_spec.bento_name, deployment_spec.bento_version
        ) as temp_path:
            ecr_image_path = create_push_image_to_ecr(
                deployment_spec.bento_name, deployment_spec.bento_version, temp_path
            )

        execution_role_arn = get_arn_role_from_current_user()
        model_name = create_sagemaker_model_name(
            deployment_spec.bento_name, deployment_spec.bento_version
        )

        sagemaker_model_info = {
            "ModelName": model_name,
            "PrimaryContainer": {
                "ContainerHostname": model_name,
                "Image": ecr_image_path,
                "Environment": {
                    "API_NAME": sagemaker_config.api_name,
                    "BENTO_SERVER_TIMEOUT": config().get(
                        'apiserver', 'default_timeout'
                    ),
                    "BENTO_SERVER_WORKERS": config().get(
                        'apiserver', 'default_gunicorn_workers_count'
                    ),
                },
            },
            "ExecutionRoleArn": execution_role_arn,
        }
        logger.info("Creating sagemaker model %s", model_name)
        create_model_response = sagemaker_client.create_model(**sagemaker_model_info)
        logger.debug("AWS create model response: %s", create_model_response)

        production_variants = [
            {
                "VariantName": generate_aws_compatible_string(
                    deployment_spec.bento_name
                ),
                "ModelName": model_name,
                "InitialInstanceCount": sagemaker_config.instance_count,
                "InstanceType": sagemaker_config.instance_type,
            }
        ]
        endpoint_config_name = create_sagemaker_endpoint_config_name(
            deployment_spec.bento_name, deployment_spec.bento_version
        )
        logger.info(
            "Creating Sagemaker endpoint %s configuration", endpoint_config_name
        )
        create_endpoint_config_response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=production_variants,
        )
        logger.debug(
            "AWS create endpoint config response: %s", create_endpoint_config_response
        )

        endpoint_name = generate_aws_compatible_string(
            deployment_pb.namespace + '-' + deployment_spec.bento_name
        )
        if prev_deployment:
            logger.info("Updating sagemaker endpoint %s", endpoint_name)
            update_endpoint_response = sagemaker_client.update_endpoint(
                EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
            )
            logger.debug("AWS update endpoint response: %s", update_endpoint_response)
        else:
            logger.info("Creating sagemaker endpoint %s", endpoint_name)
            create_endpoint_response = sagemaker_client.create_endpoint(
                EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
            )
            logger.debug("AWS create endpoint response: %s", create_endpoint_response)

        res_deployment_pb = Deployment(state=DeploymentState())
        res_deployment_pb.CopyFrom(deployment_pb)

        return ApplyDeploymentResponse(status=Status.OK(), deployment=res_deployment_pb)

    def delete(self, deployment_pb, repo=None):
        deployment_spec = deployment_pb.spec
        sagemaker_config = deployment_spec.sagemaker_operator_config
        if sagemaker_config is None:
            raise BentoMLDeploymentException('Sagemaker configuration is missing.')
        sagemaker_client = boto3.client('sagemaker', sagemaker_config.region)

        endpoint_name = generate_aws_compatible_string(
            deployment_pb.namespace + '-' + deployment_spec.bento_name
        )
        delete_endpoint_response = sagemaker_client.delete_endpoint(
            EndpointName=endpoint_name
        )
        logger.debug("AWS delete endpoint response: %s", delete_endpoint_response)
        if delete_endpoint_response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            # We will also try to delete both model and endpoint configuration for user.
            # Since they are not critical, even they failed, we will still count delete
            # deployment a success action
            model_name = create_sagemaker_model_name(
                deployment_spec.bento_name, deployment_spec.bento_version
            )
            delete_model_response = sagemaker_client.delete_model(ModelName=model_name)
            logger.debug("AWS delete model response: %s", delete_model_response)
            if delete_model_response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                logger.error(
                    "Encounter error when deleting model: %s", delete_model_response
                )

            endpoint_config_name = create_sagemaker_endpoint_config_name(
                deployment_spec.bento_name, deployment_spec.bento_version
            )
            delete_endpoint_config_response = sagemaker_client.delete_endpoint_config(  # noqa: E501
                EndpointConfigName=endpoint_config_name
            )
            logger.debug(
                "AWS delete endpoint config response: %s",
                delete_endpoint_config_response,
            )
            return DeleteDeploymentResponse(status=Status.OK())
        else:
            return DeleteDeploymentResponse(
                status=Status.INTERNAL(str(delete_endpoint_response))
            )

    def describe(self, deployment_pb, repo=None):
        deployment_spec = deployment_pb.spec
        sagemaker_config = deployment_spec.sagemaker_operator_config
        if sagemaker_config is None:
            raise BentoMLDeploymentException('Sagemaker configuration is missing.')
        sagemaker_client = boto3.client('sagemaker', sagemaker_config.region)
        endpoint_name = generate_aws_compatible_string(
            deployment_pb.namespace + '-' + deployment_spec.bento_name
        )
        endpoint_status_response = sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )
        logger.debug("AWS describe endpoint response: %s", endpoint_status_response)
        endpoint_status = endpoint_status_response["EndpointStatus"]

        service_state = ENDPOINT_STATUS_TO_STATE[endpoint_status]

        deployment_state = DeploymentState(
            state=service_state,
            info_json=json.dumps(endpoint_status_response, default=str),
        )

        return DescribeDeploymentResponse(state=deployment_state, status=Status.OK())
