import json
import logging
import os
import shutil

import boto3
from botocore.exceptions import ClientError

from bentoml.exceptions import (
    YataiDeploymentException,
    AWSServiceError,
    InvalidArgument,
    BentoMLException,
)
from bentoml.saved_bundle import loader
from bentoml.utils.tempdir import TempDirectory
from bentoml.yatai.deployment.operator import DeploymentOperatorBase
from bentoml.yatai.deployment.utils import (
    raise_if_api_names_not_found_in_bento_service_metadata,
)
from bentoml.yatai.deployment.docker_utils import (
    ensure_docker_available_or_raise,
    generate_docker_image_tag,
    push_docker_image_to_repository,
    build_docker_image,
)
from bentoml.yatai.deployment.aws_utils import (
    generate_aws_compatible_string,
    get_default_aws_region,
    get_ecr_login_info,
    create_ecr_repository_if_not_exists,
    generate_bentoml_exception_from_aws_client_error,
)
from bentoml.yatai.proto.deployment_pb2 import (
    DeploymentState,
    ApplyDeploymentResponse,
    DeleteDeploymentResponse,
    DescribeDeploymentResponse,
)
from bentoml.yatai.proto.repository_pb2 import GetBentoRequest, BentoUri
from bentoml.yatai.status import Status

logger = logging.getLogger(__name__)


BENTO_SERVICE_SAGEMAKER_DOCKERFILE = """\
FROM {docker_base_image}

# the env var $PORT is required by heroku container runtime
ENV PORT 8080
EXPOSE $PORT

RUN apt-get update --fix-missing && \
    apt-get install -y nginx && \
    apt-get clean

# gevent required by AWS Sagemaker
RUN pip install gevent>=20.9.0

# copy over model files
COPY . /bento
WORKDIR /bento

RUN if [ -f /bento/bentoml-init.sh ]; then bash -c /bento/bentoml-init.sh; fi

ENV PATH="/bento:$PATH"
"""  # noqa: E501


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
                "Service" in statement["Principal"]
                and statement["Effect"] == "Allow"
                and "sagemaker.amazonaws.com" in statement["Principal"]["Service"]
            ):
                arn = role["Arn"]
        if arn is None:
            raise YataiDeploymentException(
                "Can't find proper Arn role for Sagemaker, please create one and try "
                "again"
            )
        return arn
    elif type_role[0] in ["role", "assumed-role"]:
        role_response = iam_client.get_role(RoleName=type_role[1])
        return role_response["Role"]["Arn"]

    raise YataiDeploymentException(
        "Not supported role type {}; sts arn is {}".format(type_role[0], sts_arn)
    )


def create_and_push_docker_image_to_ecr(
    region, bento_name, bento_version, snapshot_path
):
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
    repository_id = create_ecr_repository_if_not_exists(
        region, f'{bento_name}-sagemaker'.lower()
    )
    logger.debug("Getting docker login info from AWS")
    registry_url, username, password = get_ecr_login_info(
        region, repository_id=repository_id
    )

    ecr_tag = generate_docker_image_tag(
        f'{bento_name}-sagemaker', bento_version, registry_url
    )

    logger.debug("Building docker image: %s", ecr_tag)
    build_docker_image(
        context_path=snapshot_path, dockerfile='Dockerfile-sagemaker', image_tag=ecr_tag
    )

    logger.debug("Pushing image to AWS ECR at %s", ecr_tag)
    push_docker_image_to_repository(
        repository=ecr_tag, username=username, password=password
    )
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


def _aws_client_error_to_bentoml_exception(e, message_prefix=None):
    """parse botocore.exceptions.ClientError into Bento StatusProto

    We handle two most common errors when deploying to Sagemaker.
        1. Authentication issue/invalid access(InvalidSignatureException)
        2. resources not found (ValidationException)
    It will return correlated StatusProto(NOT_FOUND, UNAUTHENTICATED)

    Args:
        e: ClientError from botocore.exceptions
    Returns:
        StatusProto
    """
    error_response = e.response.get("Error", {})
    error_code = error_response.get("Code", "Unknown")
    error_message = error_response.get("Message", "Unknown")
    error_log_message = (
        f"AWS ClientError - operation: {e.operation_name}, "
        f"code: {error_code}, message: {error_message}"
    )
    if message_prefix:
        error_log_message = f"{message_prefix}; {error_log_message}"
    logger.error(error_log_message)
    return AWSServiceError(error_log_message)


def _get_sagemaker_resource_names(deployment_pb):
    sagemaker_model_name = generate_aws_compatible_string(
        (deployment_pb.namespace, 10),
        (deployment_pb.name, 12),
        (deployment_pb.spec.bento_name, 20),
        (deployment_pb.spec.bento_version, 18),
    )
    sagemaker_endpoint_config_name = generate_aws_compatible_string(
        (deployment_pb.namespace, 10),
        (deployment_pb.name, 12),
        (deployment_pb.spec.bento_name, 20),
        (deployment_pb.spec.bento_version, 18),
    )
    sagemaker_endpoint_name = generate_aws_compatible_string(
        deployment_pb.namespace, deployment_pb.name
    )
    return sagemaker_model_name, sagemaker_endpoint_config_name, sagemaker_endpoint_name


def _delete_sagemaker_model_if_exist(sagemaker_client, sagemaker_model_name):
    try:
        delete_model_response = sagemaker_client.delete_model(
            ModelName=sagemaker_model_name
        )
        logger.debug("AWS delete model response: %s", delete_model_response)
    except ClientError as e:
        error_response = e.response.get("Error", {})
        error_code = error_response.get("Code", "Unknown")
        error_message = error_response.get("Message", "Unknown")
        if (
            error_code == "ValidationException"
            and "Could not find model" in error_message
        ):
            # sagemaker model does not exist
            return

        raise generate_bentoml_exception_from_aws_client_error(
            e, f"Failed to cleanup sagemaker model '{sagemaker_model_name}'"
        )

    return


def _delete_sagemaker_endpoint_config_if_exist(
    sagemaker_client, sagemaker_endpoint_config_name
):
    try:
        delete_endpoint_config_response = sagemaker_client.delete_endpoint_config(
            EndpointConfigName=sagemaker_endpoint_config_name
        )
        logger.debug(
            "AWS delete endpoint config response: %s", delete_endpoint_config_response
        )
    except ClientError as e:
        error_response = e.response.get("Error", {})
        error_code = error_response.get("Code", "Unknown")
        error_message = error_response.get("Message", "Unknown")
        if (
            error_code == "ValidationException"
            and "Could not find endpoint configuration" in error_message
        ):
            # endpoint config does not exist
            return

        raise generate_bentoml_exception_from_aws_client_error(
            e,
            f"Failed to cleanup sagemaker endpoint config "
            f"'{sagemaker_endpoint_config_name}' after creation failed",
        )
    return


def _delete_sagemaker_endpoint_if_exist(sagemaker_client, sagemaker_endpoint_name):
    try:
        delete_endpoint_response = sagemaker_client.delete_endpoint(
            EndpointName=sagemaker_endpoint_name
        )
        logger.debug("AWS delete endpoint response: %s", delete_endpoint_response)
    except ClientError as e:
        error_response = e.response.get("Error", {})
        error_code = error_response.get("Code", "Unknown")
        error_message = error_response.get("Message", "Unknown")
        if (
            error_code == "ValidationException"
            and "Could not find endpoint" in error_message
        ):
            # sagemaker endpoint does not exist
            return

        raise generate_bentoml_exception_from_aws_client_error(
            e, f"Failed to delete sagemaker endpoint '{sagemaker_endpoint_name}'"
        )


def delete_sagemaker_deployment_resources_if_exist(deployment_pb):
    sagemaker_config = deployment_pb.spec.sagemaker_operator_config
    sagemaker_client = boto3.client("sagemaker", sagemaker_config.region)

    (
        sagemaker_model_name,
        sagemaker_endpoint_config_name,
        sagemaker_endpoint_name,
    ) = _get_sagemaker_resource_names(deployment_pb)

    _delete_sagemaker_model_if_exist(sagemaker_client, sagemaker_model_name)
    _delete_sagemaker_endpoint_config_if_exist(
        sagemaker_client, sagemaker_endpoint_config_name
    )
    _delete_sagemaker_endpoint_if_exist(sagemaker_client, sagemaker_endpoint_name)


def _init_sagemaker_project(sagemaker_project_dir, bento_path, docker_base_image):
    shutil.copytree(bento_path, sagemaker_project_dir)

    with open(os.path.join(sagemaker_project_dir, "Dockerfile-sagemaker"), "w") as f:
        f.write(
            BENTO_SERVICE_SAGEMAKER_DOCKERFILE.format(
                docker_base_image=docker_base_image
            )
        )

    nginx_conf_path = os.path.join(os.path.dirname(__file__), "nginx.conf")
    shutil.copy(nginx_conf_path, os.path.join(sagemaker_project_dir, "nginx.conf"))

    wsgi_py_path = os.path.join(os.path.dirname(__file__), "wsgi.py")
    shutil.copy(wsgi_py_path, os.path.join(sagemaker_project_dir, "wsgi.py"))

    serve_file_path = os.path.join(os.path.dirname(__file__), "serve")
    shutil.copy(serve_file_path, os.path.join(sagemaker_project_dir, "serve"))

    # permission 755 is required for entry script 'serve'
    os.chmod(os.path.join(sagemaker_project_dir, "serve"), 0o755)
    return sagemaker_project_dir


def _create_sagemaker_model(
    sagemaker_client, sagemaker_model_name, ecr_image_path, spec
):
    execution_role_arn = get_arn_role_from_current_aws_user()

    sagemaker_model_info = {
        "ModelName": sagemaker_model_name,
        "PrimaryContainer": {
            "ContainerHostname": sagemaker_model_name,
            "Image": ecr_image_path,
            "Environment": {
                "API_NAME": spec.api_name,
                "BENTOML_GUNICORN_TIMEOUT": str(spec.timeout),
            },
        },
        "ExecutionRoleArn": execution_role_arn,
    }

    # Will set envvar, if user defined gunicorn workers per instance.  EnvVar needs
    # to be string instead of the int.
    if spec.num_of_gunicorn_workers_per_instance:
        sagemaker_model_info["PrimaryContainer"]["Environment"][
            "BENTOML_GUNICORN_NUM_OF_WORKERS"
        ] = str(spec.num_of_gunicorn_workers_per_instance)

    try:
        create_model_response = sagemaker_client.create_model(**sagemaker_model_info)
    except ClientError as e:
        raise generate_bentoml_exception_from_aws_client_error(
            e, "Failed to create sagemaker model"
        )
    logger.debug("AWS create model response: %s", create_model_response)


def _create_sagemaker_endpoint_config(
    sagemaker_client, sagemaker_model_name, endpoint_config_name, sagemaker_config
):
    create_endpoint_config_arguments = {
        "EndpointConfigName": endpoint_config_name,
        "ProductionVariants": [
            {
                "VariantName": sagemaker_model_name,
                "ModelName": sagemaker_model_name,
                "InitialInstanceCount": sagemaker_config.instance_count,
                "InstanceType": sagemaker_config.instance_type,
            }
        ],
    }

    if sagemaker_config.data_capture_s3_prefix:
        logger.debug(
            "data_capture_s3_prefix %s found, creating data capture config",
            sagemaker_config.data_capture_s3_prefix,
        )

        if not sagemaker_config.data_capture_sample_percent:
            logger.debug("data_capture_sample_percent not found, using default 100%")
            sagemaker_config.data_capture_sample_percent = 100

        create_endpoint_config_arguments["DataCaptureConfig"] = {
            "EnableCapture": True,
            "DestinationS3Uri": sagemaker_config.data_capture_s3_prefix,
            "InitialSamplingPercentage": sagemaker_config.data_capture_sample_percent,
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
        }

    logger.debug("Creating Sagemaker endpoint %s configuration", endpoint_config_name)
    try:
        create_config_response = sagemaker_client.create_endpoint_config(
            **create_endpoint_config_arguments
        )
    except ClientError as e:
        raise generate_bentoml_exception_from_aws_client_error(
            e, "Failed to create sagemaker endpoint config"
        )
    logger.debug("AWS create endpoint config response: %s", create_config_response)


def _create_sagemaker_endpoint(sagemaker_client, endpoint_name, endpoint_config_name):
    try:
        logger.debug("Creating sagemaker endpoint %s", endpoint_name)
        create_endpoint_response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        logger.debug("AWS create endpoint response: %s", create_endpoint_response)
    except ClientError as e:
        raise generate_bentoml_exception_from_aws_client_error(
            e, "Failed to create sagemaker endpoint"
        )


def _update_sagemaker_endpoint(sagemaker_client, endpoint_name, endpoint_config_name):
    try:
        logger.debug("Updating sagemaker endpoint %s", endpoint_name)
        update_endpoint_response = sagemaker_client.update_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        logger.debug("AWS update endpoint response: %s", str(update_endpoint_response))
    except ClientError as e:
        raise generate_bentoml_exception_from_aws_client_error(
            e, "Failed to update sagemaker endpoint"
        )


class SageMakerDeploymentOperator(DeploymentOperatorBase):
    def __init__(self, yatai_service):
        super(SageMakerDeploymentOperator, self).__init__(yatai_service)
        ensure_docker_available_or_raise()

    def add(self, deployment_pb):
        try:
            deployment_spec = deployment_pb.spec
            sagemaker_config = deployment_spec.sagemaker_operator_config
            sagemaker_config.region = (
                sagemaker_config.region or get_default_aws_region()
            )

            if sagemaker_config is None:
                raise YataiDeploymentException("Sagemaker configuration is missing.")

            bento_pb = self.yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            if bento_pb.bento.uri.type not in (BentoUri.LOCAL, BentoUri.S3):
                raise BentoMLException(
                    "BentoML currently not support {} repository".format(
                        BentoUri.StorageType.Name(bento_pb.bento.uri.type)
                    )
                )
            return self._add(deployment_pb, bento_pb, bento_pb.bento.uri.uri)

        except BentoMLException as error:
            deployment_pb.state.state = DeploymentState.ERROR
            deployment_pb.state.error_message = (
                f"Error creating SageMaker deployment: {str(error)}"
            )
            return ApplyDeploymentResponse(
                status=error.status_proto, deployment=deployment_pb
            )

    def _add(self, deployment_pb, bento_pb, bento_path):
        if loader._is_remote_path(bento_path):
            with loader._resolve_remote_bundle_path(bento_path) as local_path:
                return self._add(deployment_pb, bento_pb, local_path)

        deployment_spec = deployment_pb.spec
        sagemaker_config = deployment_spec.sagemaker_operator_config

        raise_if_api_names_not_found_in_bento_service_metadata(
            bento_pb.bento.bento_service_metadata, [sagemaker_config.api_name]
        )

        sagemaker_client = boto3.client("sagemaker", sagemaker_config.region)

        with TempDirectory() as temp_dir:
            sagemaker_project_dir = os.path.join(temp_dir, deployment_spec.bento_name)
            _init_sagemaker_project(
                sagemaker_project_dir,
                bento_path,
                bento_pb.bento.bento_service_metadata.env.docker_base_image,
            )
            ecr_image_path = create_and_push_docker_image_to_ecr(
                sagemaker_config.region,
                deployment_spec.bento_name,
                deployment_spec.bento_version,
                sagemaker_project_dir,
            )

        try:
            (
                sagemaker_model_name,
                sagemaker_endpoint_config_name,
                sagemaker_endpoint_name,
            ) = _get_sagemaker_resource_names(deployment_pb)

            _create_sagemaker_model(
                sagemaker_client, sagemaker_model_name, ecr_image_path, sagemaker_config
            )
            _create_sagemaker_endpoint_config(
                sagemaker_client,
                sagemaker_model_name,
                sagemaker_endpoint_config_name,
                sagemaker_config,
            )
            _create_sagemaker_endpoint(
                sagemaker_client,
                sagemaker_endpoint_name,
                sagemaker_endpoint_config_name,
            )
        except AWSServiceError as e:
            delete_sagemaker_deployment_resources_if_exist(deployment_pb)
            raise e

        return ApplyDeploymentResponse(status=Status.OK(), deployment=deployment_pb)

    def update(self, deployment_pb, previous_deployment):
        try:
            deployment_spec = deployment_pb.spec
            bento_pb = self.yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            if bento_pb.bento.uri.type not in (BentoUri.LOCAL, BentoUri.S3):
                raise BentoMLException(
                    "BentoML currently not support {} repository".format(
                        BentoUri.StorageType.Name(bento_pb.bento.uri.type)
                    )
                )
            return self._update(
                deployment_pb, previous_deployment, bento_pb, bento_pb.bento.uri.uri
            )
        except BentoMLException as error:
            deployment_pb.state.state = DeploymentState.ERROR
            deployment_pb.state.error_message = (
                f"Error updating SageMaker deployment: {str(error)}"
            )
            return ApplyDeploymentResponse(
                status=error.status_proto, deployment=deployment_pb
            )

    def _update(self, deployment_pb, current_deployment, bento_pb, bento_path):
        if loader._is_remote_path(bento_path):
            with loader._resolve_remote_bundle_path(bento_path) as local_path:
                return self._update(
                    deployment_pb, current_deployment, bento_pb, local_path
                )
        updated_deployment_spec = deployment_pb.spec
        updated_sagemaker_config = updated_deployment_spec.sagemaker_operator_config
        sagemaker_client = boto3.client(
            "sagemaker", updated_sagemaker_config.region or get_default_aws_region()
        )

        try:
            raise_if_api_names_not_found_in_bento_service_metadata(
                bento_pb.bento.bento_service_metadata,
                [updated_sagemaker_config.api_name],
            )
            describe_latest_deployment_state = self.describe(deployment_pb)
            current_deployment_spec = current_deployment.spec
            current_sagemaker_config = current_deployment_spec.sagemaker_operator_config
            latest_deployment_state = json.loads(
                describe_latest_deployment_state.state.info_json
            )

            current_ecr_image_tag = latest_deployment_state["ProductionVariants"][0][
                "DeployedImages"
            ][0]["SpecifiedImage"]
            if (
                updated_deployment_spec.bento_name != current_deployment_spec.bento_name
                or updated_deployment_spec.bento_version
                != current_deployment_spec.bento_version
            ):
                logger.debug(
                    "BentoService tag is different from current deployment, "
                    "creating new docker image and push to ECR"
                )
                with TempDirectory() as temp_dir:
                    sagemaker_project_dir = os.path.join(
                        temp_dir, updated_deployment_spec.bento_name
                    )
                    _init_sagemaker_project(
                        sagemaker_project_dir,
                        bento_path,
                        bento_pb.bento.bento_service_metadata.env.docker_base_image,
                    )
                    ecr_image_path = create_and_push_docker_image_to_ecr(
                        updated_sagemaker_config.region,
                        updated_deployment_spec.bento_name,
                        updated_deployment_spec.bento_version,
                        sagemaker_project_dir,
                    )
            else:
                logger.debug("Using existing ECR image for Sagemaker model")
                ecr_image_path = current_ecr_image_tag

            (
                updated_sagemaker_model_name,
                updated_sagemaker_endpoint_config_name,
                sagemaker_endpoint_name,
            ) = _get_sagemaker_resource_names(deployment_pb)
            (
                current_sagemaker_model_name,
                current_sagemaker_endpoint_config_name,
                _,
            ) = _get_sagemaker_resource_names(current_deployment)

            if (
                updated_sagemaker_config.api_name != current_sagemaker_config.api_name
                or updated_sagemaker_config.num_of_gunicorn_workers_per_instance
                != current_sagemaker_config.num_of_gunicorn_workers_per_instance
                or ecr_image_path != current_ecr_image_tag
            ):
                logger.debug(
                    "Sagemaker model requires update. Delete current sagemaker model %s"
                    "and creating new model %s",
                    current_sagemaker_model_name,
                    updated_sagemaker_model_name,
                )
                _delete_sagemaker_model_if_exist(
                    sagemaker_client, current_sagemaker_model_name
                )
                _create_sagemaker_model(
                    sagemaker_client,
                    updated_sagemaker_model_name,
                    ecr_image_path,
                    updated_sagemaker_config,
                )
            # When bento service tag is not changed, we need to delete the current
            # endpoint configuration in order to create new one to avoid name collation
            if (
                current_sagemaker_endpoint_config_name
                == updated_sagemaker_endpoint_config_name
            ):
                logger.debug(
                    "Current sagemaker config name %s is same as updated one, "
                    "delete it before create new endpoint config",
                    current_sagemaker_endpoint_config_name,
                )
                _delete_sagemaker_endpoint_config_if_exist(
                    sagemaker_client, current_sagemaker_endpoint_config_name
                )
            logger.debug(
                "Create new endpoint configuration %s",
                updated_sagemaker_endpoint_config_name,
            )
            _create_sagemaker_endpoint_config(
                sagemaker_client,
                updated_sagemaker_model_name,
                updated_sagemaker_endpoint_config_name,
                updated_sagemaker_config,
            )
            logger.debug(
                "Updating endpoint to new endpoint configuration %s",
                updated_sagemaker_endpoint_config_name,
            )
            _update_sagemaker_endpoint(
                sagemaker_client,
                sagemaker_endpoint_name,
                updated_sagemaker_endpoint_config_name,
            )
            if not (
                current_sagemaker_endpoint_config_name
                == updated_sagemaker_endpoint_config_name
            ):
                logger.debug(
                    'Delete old sagemaker endpoint config %s',
                    current_sagemaker_endpoint_config_name,
                )
                _delete_sagemaker_endpoint_config_if_exist(
                    sagemaker_client, current_sagemaker_endpoint_config_name
                )
        except AWSServiceError as e:
            delete_sagemaker_deployment_resources_if_exist(deployment_pb)
            raise e

        return ApplyDeploymentResponse(status=Status.OK(), deployment=deployment_pb)

    def delete(self, deployment_pb):
        try:
            deployment_spec = deployment_pb.spec
            sagemaker_config = deployment_spec.sagemaker_operator_config
            sagemaker_config.region = (
                sagemaker_config.region or get_default_aws_region()
            )
            if not sagemaker_config.region:
                raise InvalidArgument("AWS region is missing")

            delete_sagemaker_deployment_resources_if_exist(deployment_pb)

            return DeleteDeploymentResponse(status=Status.OK())
        except BentoMLException as error:
            return DeleteDeploymentResponse(status=error.status_proto)

    def describe(self, deployment_pb):
        try:
            deployment_spec = deployment_pb.spec
            sagemaker_config = deployment_spec.sagemaker_operator_config
            sagemaker_config.region = (
                sagemaker_config.region or get_default_aws_region()
            )
            if not sagemaker_config.region:
                raise InvalidArgument("AWS region is missing")
            sagemaker_client = boto3.client("sagemaker", sagemaker_config.region)
            _, _, sagemaker_endpoint_name = _get_sagemaker_resource_names(deployment_pb)

            try:
                endpoint_status_response = sagemaker_client.describe_endpoint(
                    EndpointName=sagemaker_endpoint_name
                )
            except ClientError as e:
                raise _aws_client_error_to_bentoml_exception(
                    e,
                    f"Failed to fetch current status of sagemaker endpoint "
                    f"'{sagemaker_endpoint_name}'",
                )

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
            return DescribeDeploymentResponse(status=error.status_proto)
