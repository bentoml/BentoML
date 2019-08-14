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

from bentoml.config import config
from bentoml.deployment.legacy_deployment import LegacyDeployment
from bentoml.deployment.utils import (
    generate_bentoml_deployment_snapshot_path,
    process_docker_api_line,
)
from bentoml.yatai.status import Status
from bentoml.utils.whichcraft import which
from bentoml.utils.tempdir import TempDirectory
from bentoml.exceptions import BentoMLException, BentoMLDeploymentException
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
    return generate_aws_compatible_string("bentoml-" + bento_name + "-" + bento_version)


def create_sagemaker_endpoint_config_name(bento_name, bento_version):
    return generate_aws_compatible_string(
        bento_name + "-" + bento_version + "-configuration"
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


def generate_sagemaker_snapshot(name, version, archive_path):
    snapshot_path = generate_bentoml_deployment_snapshot_path(
        name, version, "aws-sagemaker"
    )
    shutil.copytree(archive_path, snapshot_path)
    with open(os.path.join(snapshot_path, "nginx.conf"), "w") as f:
        f.write(DEFAULT_NGINX_CONFIG)
    with open(os.path.join(snapshot_path, "wsgi.py"), "w") as f:
        f.write(DEFAULT_WSGI_PY)
    with open(os.path.join(snapshot_path, "serve"), "w") as f:
        f.write(DEFAULT_SERVE_SCRIPT)

    # permission 755 is required for entry script 'serve'
    permission = "755"
    octal_permission = int(permission, 8)
    os.chmod(os.path.join(snapshot_path, "serve"), octal_permission)
    return snapshot_path


class SagemakerDeployment(LegacyDeployment):
    def __init__(
        self,
        archive_path,
        api_name,
        region=None,
        instance_count=None,
        instance_type=None,
    ):
        if which("docker") is None:
            raise ValueError(
                "docker is not installed, please install docker and then try again"
            )
        super(SagemakerDeployment, self).__init__(archive_path)
        self.region = DEFAULT_REGION if region is None else region
        self.instance_count = (
            DEFAULT_INSTANCE_COUNT if instance_count is None else instance_count
        )
        self.instance_type = (
            DEFAULT_INSTANCE_TYPE if instance_type is None else instance_type
        )
        self.api = self.bento_service.get_service_api(api_name)
        self.sagemaker_client = boto3.client("sagemaker", region_name=self.region)
        self.model_name = generate_aws_compatible_string(
            "bentoml-" + self.bento_service.name + "-" + self.bento_service.version
        )
        self.endpoint_config_name = generate_aws_compatible_string(
            self.bento_service.name
            + "-"
            + self.bento_service.version
            + "-configuration"
        )

    def deploy(self):
        """Deploy BentoML service to AWS Sagemaker.
        Your AWS credential must have the correct permissions for sagemaker and ECR

        1. generate snapshot for aws sagemaker
        2. Create docker image and push to ECR
        3. Create sagemaker model base on the ECR image
        4. Create sagemaker endpoint configuration base on sagemaker model
        5. Create sagemaker endpoint base on sagemaker endpoint configuration

        Args:
            archive_path(Path, str)
            additional_info(dict)

        Returns:
            str: Location to the output snapshot's path
        """
        snapshot_path = generate_sagemaker_snapshot(
            self.bento_service.name, self.bento_service.version, self.archive_path
        )

        execution_role_arn = get_arn_role_from_current_user()
        ecr_image_path = create_push_image_to_ecr(
            self.bento_service.name, self.bento_service.version, snapshot_path
        )
        sagemaker_model_info = {
            "ModelName": self.model_name,
            "PrimaryContainer": {
                "ContainerHostname": self.model_name,
                "Image": ecr_image_path,
                "Environment": {"API_NAME": self.api.name},
            },
            "ExecutionRoleArn": execution_role_arn,
        }
        logger.info("Creating sagemaker model %s", self.model_name)
        create_model_response = self.sagemaker_client.create_model(
            **sagemaker_model_info
        )
        logger.info("AWS create model response: %s", create_model_response)

        production_variants = [
            {
                "VariantName": self.bento_service.name,
                "ModelName": self.model_name,
                "InitialInstanceCount": self.instance_count,
                "InstanceType": self.instance_type,
            }
        ]
        logger.info(
            "Creating sagemaker endpoint %s configuration", self.endpoint_config_name
        )
        create_endpoint_config_response = self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=self.endpoint_config_name,
            ProductionVariants=production_variants,
        )
        logger.info(
            "AWS create endpoint config response: %s", create_endpoint_config_response
        )

        logger.info("Creating sagemaker endpoint %s", self.bento_service.name)
        create_endpoint_response = self.sagemaker_client.create_endpoint(
            EndpointName=self.bento_service.name,
            EndpointConfigName=self.endpoint_config_name,
        )
        logger.info("AWS create endpoint response: %s", create_endpoint_response)

        # TODO: maybe wait for this endpoint from creating to running and then return
        return snapshot_path

    def check_status(self):
        endpoint_status_response = self.sagemaker_client.describe_endpoint(
            EndpointName=self.bento_service.name
        )
        logger.info("AWS describe endpoint response: %s", endpoint_status_response)
        endpoint_in_service = endpoint_status_response["EndpointStatus"] == "InService"

        status_message = "{service} is {status}".format(
            service=self.bento_service.name,
            status=endpoint_status_response["EndpointStatus"],
        )
        if endpoint_in_service:
            status_message += (
                "\nEndpoint ARN: " + endpoint_status_response["EndpointArn"]
            )

        return endpoint_in_service, status_message

    def delete(self):
        """Delete Sagemaker endpoint for the bentoml service.
        It will also delete the model or the endpoint configuration.

        return: Boolean, True if successful
        """
        if not self.check_status()[0]:
            raise BentoMLException(
                "No active AWS Sagemaker deployment for service %s"
                % self.bento_service.name
            )

        delete_endpoint_response = self.sagemaker_client.delete_endpoint(
            EndpointName=self.bento_service.name
        )
        logger.info("AWS delete endpoint response: %s", delete_endpoint_response)
        if delete_endpoint_response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            # We will also try to delete both model and endpoint configuration for user.
            # Since they are not critical, even they failed, we will still count delete
            # deployment a success action
            delete_model_response = self.sagemaker_client.delete_model(
                ModelName=self.model_name
            )
            logger.info("AWS delete model response: %s", delete_model_response)
            if delete_model_response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                logger.error(
                    "Encounter error when deleting model: %s", delete_model_response
                )

            delete_endpoint_config_response = self.sagemaker_client.delete_endpoint_config(  # noqa: E501
                EndpointConfigName=self.endpoint_config_name
            )
            logger.info(
                "AWS delete endpoint config response: %s",
                delete_endpoint_config_response,
            )
            if (
                delete_endpoint_config_response["ResponseMetadata"]["HTTPStatusCode"]
                != 200
            ):
                logger.error(
                    "Encounter error when deleting endpoint configuration: %s",
                    delete_endpoint_config_response,
                )
            return True
        else:
            return False


# Deployment Service MVP Working-In-Progress
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

    def cleanup(self, ignore_errors=False):
        self.temp_directory.cleanup()
        self.path = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cleanup:
            self.cleanup()


class SageMakerDeploymentOperator(DeploymentOperatorBase):
    def apply(self, deployment_pb, repo):
        deployment_spec = deployment_pb.spec
        sagemaker_config = deployment_spec.sagemaker_operator_config
        if sagemaker_config is None:
            raise BentoMLDeploymentException('Sagemaker configuration is missing.')

        if sagemaker_config.region is not None:
            region = sagemaker_config.region
        else:
            region = config.get('aws', 'default_region')

        archive_path = repo.get(deployment_spec.bento_name, deployment_spec.bento_version)

        # config = load_bentoml_config(bento_path)...

        sagemaker_client = boto3.client('sagemaker', region)

        temp_path = generate_temporary_sagemaker_content(archive_path)
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
                "Environment": {"API_NAME": sagemaker_config.api_name},
            },
            "ExecutionRoleArn": execution_role_arn,
        }
        logger.info("Creating sagemaker model %s", model_name)
        create_model_response = sagemaker_client.create_model(**sagemaker_model_info)
        logger.info("AWS create model response: %s", create_model_response)

        production_variants = [
            {
                "VariantName": generate_aws_compatible_string(deployment_spec.bento_name),
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
        logger.info(
            "AWS create endpoint config response: %s", create_endpoint_config_response
        )

        logger.info("Creating sagemaker endpoint %s", deployment_spec.bento_name)
        create_endpoint_response = sagemaker_client.create_endpoint(
            EndpointName=generate_aws_compatible_string(deployment_spec.bento_name),
            EndpointConfigName=endpoint_config_name,
        )
        logger.info("AWS create endpoint response: %s", create_endpoint_response)

        # TODO wait for the sagemaker status from creating to running

        res_deployment_pb = Deployment()
        res_deployment_pb.CopyFrom(deployment_pb)
        res_deployment_pb.state = self.describe(res_deployment_pb)
        
        return ApplyDeploymentResponse(
            status=Status.OK(),
            deployment=res_deployment_pb
        )

    def delete(self, deployment_pb, repo):
        deployment = self.get(deployment_pb).deployment
        deployment_spec = deployment.spec
        sagemaker_config = deployment_spec.sagemaker_operator_config
        if sagemaker_config is None:
            raise BentoMLDeploymentException('Sagemaker configuration is missing.')
        sagemaker_client = boto3.client('sagemaker', sagemaker_config.region)

        delete_endpoint_response = sagemaker_client.delete_endpoint(
            EndpointName=deployment_spec.bento_name
        )
        logger.info("AWS delete endpoint response: %s", delete_endpoint_response)
        if delete_endpoint_response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            # We will also try to delete both model and endpoint configuration for user.
            # Since they are not critical, even they failed, we will still count delete
            # deployment a success action
            model_name = create_sagemaker_model_name(
                deployment_spec.bento_name, deployment_spec.bento_version
            )
            delete_model_response = sagemaker_client.delete_model(ModelName=model_name)
            logger.info("AWS delete model response: %s", delete_model_response)
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
            logger.info(
                "AWS delete endpoint config response: %s",
                delete_endpoint_config_response,
            )
            return DeleteDeploymentResponse(status=Status.OK())
        else:
            raise BentoMLDeploymentException('Delete Sagemaker deployment failed')

    def describe(self, deployment_pb, repo):
        deployment_spec = deployment_pb.spec
        sagemaker_config = deployment_spec.sagemaker_operator_config
        if sagemaker_config is None:
            raise BentoMLDeploymentException('Sagemaker configuration is missing.')
        sagemaker_client = boto3.client('sagemaker', sagemaker_config.region)
        endpoint_status_response = sagemaker_client.describe_endpoint(
            EndpointName=deployment_spec.bento_name
        )
        logger.info("AWS describe endpoint response: %s", endpoint_status_response)
        endpoint_status = endpoint_status_response["EndpointStatus"]

        # Sagemaker response status: 'OutOfService'|'Creating'|'Updating'|
        #                            'SystemUpdating'|'RollingBack'|'InService'|
        #                            'Deleting'|'Failed'
        if endpoint_status == 'InService':
            service_state = DeploymentState.RUNNING
        elif endpoint_status in [
            'Creating',
            'Updating',
            'RollingBack',
            'SystemUpdating',
        ]:
            service_state = DeploymentState.PENDING
        elif endpoint_status == 'Deleting':
            service_state = DeploymentState.INACTIVATED
        elif endpoint_status == 'Failed':
            service_state = DeploymentState.ERROR
        else:
            service_state = DeploymentState.ERROR

        info_json = {
            "endpoint_status": endpoint_status,
            "arn": endpoint_status_response["EndpointArn"],
            "ProductionVariants": endpoint_status_response["ProductionVariants"],
        }
        deployment_state = DeploymentState(
            state=service_state,
            info_json=json.dumps(info_json),
            error_message=endpoint_status_response["FailureReason"],
        )
        return deployment_state
