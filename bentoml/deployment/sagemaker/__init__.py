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
import json

from six.moves.urllib.parse import urlparse
import boto3
import docker

from botocore.exceptions import ClientError

from bentoml import config
from bentoml.bundler import loader
from bentoml.deployment.utils import (
    process_docker_api_line,
    ensure_docker_available_or_raise,
    exception_to_return_status,
    generate_aws_compatible_string,
    raise_if_api_names_not_found_in_bento_service_metadata,
)
from bentoml.proto.repository_pb2 import GetBentoRequest, BentoUri
from bentoml.yatai.status import Status
from bentoml.utils.tempdir import TempDirectory
from bentoml.exceptions import BentoMLDeploymentException, BentoMLException
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

# This should be kept in sync with BENTO_SERVICE_DOCKERFILE_CPU_TEMPLATE
BENTO_SERVICE_SAGEMAKER_DOCKERFILE = """\
FROM continuumio/miniconda3:4.7.12

EXPOSE 8080

RUN set -x \\
     && apt-get update \\
     && apt-get install --no-install-recommends --no-install-suggests -y libpq-dev build-essential\\
     && apt-get install -y nginx \\
     && rm -rf /var/lib/apt/lists/*

# update conda, pre-install BentoML base dependencies
RUN conda update conda -y \\
      && conda install pip numpy scipy \\
      && pip install gunicorn six gevent

# copy over model files
COPY . /opt/program
WORKDIR /opt/program

# update conda base env
RUN conda env update -n base -f /opt/program/environment.yml
RUN pip install -r /opt/program/requirements.txt

# Install additional pip dependencies inside bundled_pip_dependencies dir
RUN if [ -f /bento/bentoml_init.sh ]; then /bin/bash -c /bento/bentoml_init.sh; fi

# run user defined setup script
RUN if [ -f /opt/program/setup.sh ]; then /bin/bash -c /opt/program/setup.sh; fi

ENV PATH="/opt/program:${PATH}"
"""  # noqa: E501


def strip_scheme(url):
    """ Stripe url's schema
    e.g.   http://some.url/path -> some.url/path
    :param url: String
    :return: String
    """
    parsed = urlparse(url)
    scheme = "%s://" % parsed.scheme
    return parsed.geturl().replace(scheme, "", 1)


def get_arn_role_from_current_aws_user():
    sts_client = boto3.client("sts")
    identity = sts_client.get_caller_identity()
    sts_arn = identity["Arn"]
    sts_arn_list = sts_arn.split(":")
    type_role = sts_arn_list[-1].split("/")
    iam_client = boto3.client("iam")
    if type_role[0] in ("user", "root"):
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
            raise BentoMLDeploymentException(
                "Can't find proper Arn role for Sagemaker, please create one and try "
                "again"
            )
        return arn
    elif type_role[0] == "role":
        role_response = iam_client.get_role(RoleName=type_role[1])
        return role_response["Role"]["Arn"]

    raise BentoMLDeploymentException(
        "Not supported role type {}; sts arn is {}".format(type_role[0], sts_arn)
    )


def create_push_docker_image_to_ecr(region, bento_name, bento_version, snapshot_path):
    """Create BentoService sagemaker image and push to AWS ECR

    Example: https://github.com/awslabs/amazon-sagemaker-examples/blob/\
        master/advanced_functionality/scikit_bring_your_own/container/build_and_push.sh
    1. get aws account info and login ecr
    2. create ecr repository, if not exist
    3. build tag and push docker image

    Args:
        region(String)
        bento_name(String)
        bento_version(String)
        snapshot_path(Path)

    Returns:
        str: AWS ECR Tag
    """
    ecr_client = boto3.client("ecr", region)
    token = ecr_client.get_authorization_token()
    logger.debug("Getting docker login info from AWS")
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

    logger.debug("Building docker image: %s", ecr_tag)
    for line in docker_api.build(
        path=snapshot_path, dockerfile="Dockerfile-sagemaker", tag=ecr_tag
    ):
        process_docker_api_line(line)

    try:
        ecr_client.describe_repositories(repositoryNames=[image_name])["repositories"]
    except ecr_client.exceptions.RepositoryNotFoundException:
        ecr_client.create_repository(repositoryName=image_name)

    logger.debug("Pushing image to AWS ECR at %s", ecr_tag)
    for line in docker_api.push(ecr_tag, stream=True, auth_config=auth_config_payload):
        process_docker_api_line(line)
    logger.debug("Finished pushing image: %s", ecr_tag)
    return ecr_tag


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


def _parse_aws_client_exception(e):
    """parse botocore.exceptions.ClientError into Bento StatusProto

    We handle two most common errors when deploying to Sagemaker.
        1. Authenication issue/invalid access(InvalidSignatureException)
        2. resources not found (ValidationException)
    It will return correlated StatusProto(NOT_FOUND, UNAUTHENTICATED)

    Args:
        e: ClientError from botocore.exceptions
    Returns:
        StatusProto
    """
    error_response = e.response.get('Error', {})
    error_code = error_response.get('Code')
    error_message = error_response.get('Message', 'Unknown')
    error_log_message = 'AWS ClientError for {operation}: {code} - {message}'.format(
        operation=e.operation_name, code=error_code, message=error_message
    )
    logger.error(error_log_message)
    if error_code == 'ValidationException':
        return Status.NOT_FOUND(error_log_message)
    elif error_code == 'InvalidSignatureException':
        return Status.UNAUTHENTICATED(error_log_message)
    else:
        return Status.INTERNAL('Error interacting with AWS service {}'.format(str(e)))


def _cleanup_sagemaker_model(client, name, version):
    model_name = generate_aws_compatible_string(name + '-' + version)
    try:
        delete_model_response = client.delete_model(ModelName=model_name)
        logger.debug("AWS delete model response: %s", delete_model_response)
    except ClientError as e:
        return _parse_aws_client_exception(e)

    return


def _cleanup_sagemaker_endpoint_config(client, name, version):
    endpoint_config_name = generate_aws_compatible_string(name + '-' + version)
    try:
        delete_endpoint_config_response = client.delete_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        logger.debug(
            "AWS delete endpoint config response: %s", delete_endpoint_config_response
        )
    except ClientError as e:
        return _parse_aws_client_exception(e)
    return


def init_sagemaker_project(sagemaker_project_dir, bento_path):
    shutil.copytree(bento_path, sagemaker_project_dir)

    with open(os.path.join(sagemaker_project_dir, 'Dockerfile-sagemaker'), "w") as f:
        f.write(BENTO_SERVICE_SAGEMAKER_DOCKERFILE)
    with open(os.path.join(sagemaker_project_dir, "nginx.conf"), "w") as f:
        f.write(DEFAULT_NGINX_CONFIG)
    with open(os.path.join(sagemaker_project_dir, "wsgi.py"), "w") as f:
        f.write(DEFAULT_WSGI_PY)
    with open(os.path.join(sagemaker_project_dir, "serve"), "w") as f:
        f.write(DEFAULT_SERVE_SCRIPT)

    # permission 755 is required for entry script 'serve'
    permission = "755"
    octal_permission = int(permission, 8)
    os.chmod(os.path.join(sagemaker_project_dir, "serve"), octal_permission)
    return sagemaker_project_dir


def _create_sagemaker_model(
    sagemaker_client, bento_name, bento_version, ecr_image_path, api_name
):
    execution_role_arn = get_arn_role_from_current_aws_user()
    model_name = generate_aws_compatible_string(bento_name + '-' + bento_version)

    sagemaker_model_info = {
        "ModelName": model_name,
        "PrimaryContainer": {
            "ContainerHostname": model_name,
            "Image": ecr_image_path,
            "Environment": {
                "API_NAME": api_name,
                "BENTO_SERVER_TIMEOUT": config().get('apiserver', 'default_timeout'),
                "BENTO_SERVER_WORKERS": config().get(
                    'apiserver', 'default_gunicorn_workers_count'
                ),
            },
        },
        "ExecutionRoleArn": execution_role_arn,
    }

    logger.debug("Creating sagemaker model %s", model_name)
    create_model_response = sagemaker_client.create_model(**sagemaker_model_info)
    logger.debug("AWS create model response: %s", create_model_response)
    return model_name


def _create_sagemaker_endpoint_config(
    sagemaker_client, model_name, bento_name, bento_version, sagemaker_config
):
    production_variants = [
        {
            "VariantName": generate_aws_compatible_string(bento_name),
            "ModelName": model_name,
            "InitialInstanceCount": sagemaker_config.instance_count,
            "InstanceType": sagemaker_config.instance_type,
        }
    ]
    endpoint_config_name = generate_aws_compatible_string(
        bento_name + '-' + bento_version
    )

    logger.debug("Creating Sagemaker endpoint %s configuration", endpoint_config_name)
    create_config_response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name, ProductionVariants=production_variants
    )
    logger.debug("AWS create endpoint config response: %s", create_config_response)
    return endpoint_config_name


class SageMakerDeploymentOperator(DeploymentOperatorBase):
    def apply(self, deployment_pb, yatai_service, prev_deployment=None):
        try:
            ensure_docker_available_or_raise()
            deployment_spec = deployment_pb.spec
            sagemaker_config = deployment_spec.sagemaker_operator_config
            if sagemaker_config is None:
                raise BentoMLDeploymentException('Sagemaker configuration is missing.')

            bento_pb = yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            if bento_pb.bento.uri.type not in (BentoUri.LOCAL, BentoUri.S3):
                raise BentoMLException(
                    'BentoML currently not support {} repository'.format(
                        bento_pb.bento.uri.type
                    )
                )

            return self._apply(
                deployment_pb,
                bento_pb,
                yatai_service,
                bento_pb.bento.uri.uri,
                prev_deployment,
            )

        except BentoMLException as error:
            return ApplyDeploymentResponse(status=exception_to_return_status(error))

    def _apply(
        self, deployment_pb, bento_pb, yatai_service, bento_path, prev_deployment=None
    ):
        if loader._is_remote_path(bento_path):
            with loader._resolve_remote_bundle_path(bento_path) as local_path:
                return self._apply(
                    deployment_pb, bento_pb, yatai_service, local_path, prev_deployment
                )

        deployment_spec = deployment_pb.spec
        sagemaker_config = deployment_spec.sagemaker_operator_config

        raise_if_api_names_not_found_in_bento_service_metadata(
            bento_pb.bento.bento_service_metadata, [sagemaker_config.api_name]
        )

        sagemaker_client = boto3.client('sagemaker', sagemaker_config.region)

        with TempDirectory() as temp_dir:
            sagemaker_project_dir = os.path.join(temp_dir, deployment_spec.bento_name)
            init_sagemaker_project(sagemaker_project_dir, bento_path)
            ecr_image_path = create_push_docker_image_to_ecr(
                sagemaker_config.region,
                deployment_spec.bento_name,
                deployment_spec.bento_version,
                sagemaker_project_dir,
            )

        try:
            model_name = _create_sagemaker_model(
                sagemaker_client,
                deployment_spec.bento_name,
                deployment_spec.bento_version,
                ecr_image_path,
                sagemaker_config.api_name,
            )
        except ClientError as e:
            status = _parse_aws_client_exception(e)
            status.error_message = (
                'Failed to create model for SageMaker'
                ' Deployment: {}'.format(status.error_message)
            )
            return ApplyDeploymentResponse(status=status, deployment=deployment_pb)

        try:
            endpoint_config_name = _create_sagemaker_endpoint_config(
                sagemaker_client,
                model_name,
                deployment_spec.bento_name,
                deployment_spec.bento_version,
                sagemaker_config,
            )
        except ClientError as e:
            # create endpoint failed, will remove previously created model
            cleanup_model_error = _cleanup_sagemaker_model(
                sagemaker_client,
                deployment_spec.bento_name,
                deployment_spec.bento_version,
            )
            if cleanup_model_error:
                cleanup_model_error.error_message = (
                    'Failed to clean up model after unsuccessfully '
                    'create endpoint config: %s',
                    cleanup_model_error.error_message,
                )
                return ApplyDeploymentResponse(
                    status=cleanup_model_error, deployment=deployment_pb
                )

            status = _parse_aws_client_exception(e)
            status.error_message = (
                'Failed to create endpoint config for SageMaker deployment: %s',
                status.error_message,
            )
            return ApplyDeploymentResponse(status=status, deployment=deployment_pb)

        endpoint_name = generate_aws_compatible_string(
            deployment_pb.namespace + '-' + deployment_spec.bento_name
        )
        try:
            if prev_deployment:
                logger.debug("Updating sagemaker endpoint %s", endpoint_name)
                update_endpoint_response = sagemaker_client.update_endpoint(
                    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
                )
                logger.debug(
                    "AWS update endpoint response: %s", update_endpoint_response
                )
            else:
                logger.debug("Creating sagemaker endpoint %s", endpoint_name)
                create_endpoint_response = sagemaker_client.create_endpoint(
                    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
                )
                logger.debug(
                    "AWS create endpoint response: %s", create_endpoint_response
                )
        except ClientError as e:
            # create/update endpoint failed, will remove previously created config
            # and then remove the model
            cleanup_endpoint_config_error = _cleanup_sagemaker_endpoint_config(
                client=sagemaker_client,
                name=deployment_spec.bento_name,
                version=deployment_spec.bento_version,
            )
            if cleanup_endpoint_config_error:
                cleanup_endpoint_config_error.error_message = (
                    'Failed to clean up endpoint config after unsuccessfully '
                    'apply SageMaker deployment: %s',
                    cleanup_endpoint_config_error.error_message,
                )
                return ApplyDeploymentResponse(
                    status=cleanup_endpoint_config_error, deployment=deployment_pb
                )

            cleanup_model_error = _cleanup_sagemaker_model(
                client=sagemaker_client,
                name=deployment_spec.bento_name,
                version=deployment_spec.bento_version,
            )
            if cleanup_model_error:
                cleanup_model_error.error_message = (
                    'Failed to clean up model after unsuccessfully apply '
                    'SageMaker deployment: {}'.format(cleanup_model_error.error_message)
                )
                return ApplyDeploymentResponse(
                    status=cleanup_model_error, deployment=deployment_pb
                )

            status = _parse_aws_client_exception(e)
            status.error_message = 'Failed to apply SageMaker ' 'deployment: {}'.format(
                status.error_message
            )
            return ApplyDeploymentResponse(status=status, deployment=deployment_pb)

        res_deployment_pb = Deployment(state=DeploymentState())
        res_deployment_pb.CopyFrom(deployment_pb)

        return ApplyDeploymentResponse(status=Status.OK(), deployment=res_deployment_pb)

    def delete(self, deployment_pb, yatai_service=None):
        try:
            deployment_spec = deployment_pb.spec
            sagemaker_config = deployment_spec.sagemaker_operator_config
            if sagemaker_config is None:
                raise BentoMLDeploymentException('Sagemaker configuration is missing.')
            sagemaker_client = boto3.client('sagemaker', sagemaker_config.region)

            endpoint_name = generate_aws_compatible_string(
                deployment_pb.namespace + '-' + deployment_spec.bento_name
            )
            try:
                delete_endpoint_response = sagemaker_client.delete_endpoint(
                    EndpointName=endpoint_name
                )
                logger.debug(
                    "AWS delete endpoint response: %s", delete_endpoint_response
                )
            except ClientError as e:
                status = _parse_aws_client_exception(e)
                status.error_message = 'Failed to delete SageMaker endpoint: {}'.format(
                    status.error_message
                )
                return DeleteDeploymentResponse(status=status)

            delete_config_error = _cleanup_sagemaker_endpoint_config(
                client=sagemaker_client,
                name=deployment_spec.bento_name,
                version=deployment_spec.bento_version,
            )
            if delete_config_error:
                delete_config_error.error_message = (
                    'Failed to delete SageMaker '
                    'endpoint config: {}'.format(delete_config_error.error_message)
                )
                return DeleteDeploymentResponse(status=delete_config_error)

            delete_model_error = _cleanup_sagemaker_model(
                client=sagemaker_client,
                name=deployment_spec.bento_name,
                version=deployment_spec.bento_version,
            )
            if delete_model_error:
                delete_model_error.error_message = (
                    'Failed to delete SageMaker '
                    'model: {}'.format(delete_model_error.error_message)
                )
                return DeleteDeploymentResponse(status=delete_model_error)

            return DeleteDeploymentResponse(status=Status.OK())
        except BentoMLException as error:
            return DeleteDeploymentResponse(status=exception_to_return_status(error))

    def describe(self, deployment_pb, yatai_service=None):
        try:
            deployment_spec = deployment_pb.spec
            sagemaker_config = deployment_spec.sagemaker_operator_config
            if sagemaker_config is None:
                raise BentoMLDeploymentException('Sagemaker configuration is missing.')
            sagemaker_client = boto3.client('sagemaker', sagemaker_config.region)
            endpoint_name = generate_aws_compatible_string(
                deployment_pb.namespace + '-' + deployment_spec.bento_name
            )
            try:
                endpoint_status_response = sagemaker_client.describe_endpoint(
                    EndpointName=endpoint_name
                )
            except ClientError as e:
                status = _parse_aws_client_exception(e)
                status.error_message = (
                    'Failed to describe SageMaker '
                    'deployment: {}'.format(status.error_message)
                )
                return DescribeDeploymentResponse(status=status)

            logger.debug("AWS describe endpoint response: %s", endpoint_status_response)
            endpoint_status = endpoint_status_response["EndpointStatus"]

            service_state = ENDPOINT_STATUS_TO_STATE[endpoint_status]

            deployment_state = DeploymentState(
                state=service_state,
                info_json=json.dumps(endpoint_status_response, default=str),
            )
            deployment_state.timestamp.GetCurrentTime()

            return DescribeDeploymentResponse(
                state=deployment_state, status=Status.OK()
            )
        except BentoMLException as error:
            return DescribeDeploymentResponse(status=exception_to_return_status(error))
