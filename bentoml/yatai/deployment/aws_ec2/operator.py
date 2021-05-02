import os
import base64
import json
import logging

from bentoml.utils.tempdir import TempDirectory

from bentoml.utils.lazy_loader import LazyLoader
from bentoml.saved_bundle import loader
from bentoml.yatai.deployment.aws_ec2.templates import (
    EC2_CLOUDFORMATION_TEMPLATE,
    EC2_USER_INIT_SCRIPT,
)
from bentoml.yatai.deployment.operator import DeploymentOperatorBase
from bentoml.yatai.proto.deployment_pb2 import (
    ApplyDeploymentResponse,
    DeleteDeploymentResponse,
    DescribeDeploymentResponse,
    DeploymentState,
)
from bentoml.yatai.status import Status
from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.utils.s3 import create_s3_bucket_if_not_exists
from bentoml.yatai.deployment.docker_utils import (
    ensure_docker_available_or_raise,
    generate_docker_image_tag,
    build_docker_image,
    push_docker_image_to_repository,
)
from bentoml.yatai.deployment.aws_utils import (
    generate_aws_compatible_string,
    get_default_aws_region,
    ensure_sam_available_or_raise,
    validate_sam_template,
    FAILED_CLOUDFORMATION_STACK_STATUS,
    cleanup_s3_bucket_if_exist,
    delete_cloudformation_stack,
    delete_ecr_repository,
    get_instance_ip_from_scaling_group,
    get_aws_user_id,
    create_ecr_repository_if_not_exists,
    get_ecr_login_info,
    describe_cloudformation_stack,
)
from bentoml.exceptions import (
    BentoMLException,
    InvalidArgument,
    YataiDeploymentException,
)
from bentoml.yatai.proto.repository_pb2 import GetBentoRequest, BentoUri
from bentoml.yatai.proto import status_pb2
from bentoml.yatai.deployment.aws_ec2.utils import (
    build_template,
    package_template,
    deploy_template,
    get_endpoints_from_instance_address,
    get_healthy_target,
)
from bentoml.yatai.deployment.aws_ec2.constants import (
    BENTOSERVICE_PORT,
    TARGET_HEALTH_CHECK_INTERVAL,
    TARGET_HEALTH_CHECK_PATH,
    TARGET_HEALTH_CHECK_PORT,
    TARGET_HEALTH_CHECK_TIMEOUT_SECONDS,
    TARGET_HEALTH_CHECK_THRESHOLD_COUNT,
)

logger = logging.getLogger(__name__)

yatai_proto = LazyLoader("yatai_proto", globals(), "bentoml.yatai.proto")
SAM_TEMPLATE_NAME = "template.yml"


def _make_user_data(registry, tag, region):
    """
    Create init script for EC2 containers to download docker image,
    and run container on startup.
    args:
        registry: ECR registry domain
        tag: bento tag
        region: AWS region
    """

    base_format = EC2_USER_INIT_SCRIPT.format(
        registry=registry, tag=tag, region=region, bentoservice_port=BENTOSERVICE_PORT
    )
    encoded = base64.b64encode(base_format.encode("ascii")).decode("ascii")
    return encoded


def _make_cloudformation_template(
    project_dir,
    user_data,
    s3_bucket_name,
    sam_template_name,
    elb_name,
    ami_id,
    instance_type,
    autoscaling_min_size,
    autoscaling_desired_capacity,
    autoscaling_max_size,
):
    """
    Create and save cloudformation template for deployment
    args:
        project_dir: path to save template file
        user_data: base64 encoded user data for cloud-init script
        s3_bucket_name: AWS S3 bucket name
        sam_template_name: template name to save
        ami_id: ami id for EC2 container to use
        instance_type: EC2 instance type
        autocaling_min_size: autoscaling group minimum size
        autocaling_desired_capacity: autoscaling group desired size
        autocaling_min_size: autoscaling group maximum size


    NOTE: SSH ACCESS TO INSTANCE MAY NOT BE REQUIRED
    TODO: Port taken from cli
    """

    template_file_path = os.path.join(project_dir, sam_template_name)
    with open(template_file_path, "a") as f:
        f.write(
            EC2_CLOUDFORMATION_TEMPLATE.format(
                ami_id=ami_id,
                template_name=sam_template_name,
                instance_type=instance_type,
                user_data=user_data,
                elb_name=elb_name,
                autoscaling_min_size=autoscaling_min_size,
                autoscaling_desired_capacity=autoscaling_desired_capacity,
                autoscaling_max_size=autoscaling_max_size,
                s3_bucket_name=s3_bucket_name,
                target_health_check_interval_seconds=TARGET_HEALTH_CHECK_INTERVAL,
                target_health_check_path=TARGET_HEALTH_CHECK_PATH,
                target_health_check_port=TARGET_HEALTH_CHECK_PORT,
                target_health_check_timeout_seconds=TARGET_HEALTH_CHECK_TIMEOUT_SECONDS,
                target_health_check_threshold_count=TARGET_HEALTH_CHECK_THRESHOLD_COUNT,
            )
        )
    return template_file_path


def generate_ec2_resource_names(namespace, name):
    sam_template_name = generate_aws_compatible_string(
        f"btml-template-{namespace}-{name}"
    )
    deployment_stack_name = generate_aws_compatible_string(
        f"btml-stack-{namespace}-{name}"
    )
    repo_name = generate_aws_compatible_string(f"btml-repo-{namespace}-{name}")
    elb_name = generate_aws_compatible_string(f"{namespace}-{name}", max_length=32)

    return sam_template_name, deployment_stack_name, repo_name, elb_name


def deploy_ec2_service(
    deployment_pb,
    deployment_spec,
    bento_path,
    aws_ec2_deployment_config,
    s3_bucket_name,
    region,
):
    (
        sam_template_name,
        deployment_stack_name,
        repo_name,
        elb_name,
    ) = generate_ec2_resource_names(deployment_pb.namespace, deployment_pb.name)

    with TempDirectory() as project_path:
        repository_id = create_ecr_repository_if_not_exists(region, repo_name)
        registry_url, username, password = get_ecr_login_info(region, repository_id)
        ecr_tag = generate_docker_image_tag(
            repo_name, deployment_spec.bento_version, registry_url
        )
        build_docker_image(context_path=bento_path, image_tag=ecr_tag)
        push_docker_image_to_repository(
            repository=ecr_tag, username=username, password=password
        )

        logger.info("Generating user data")
        encoded_user_data = _make_user_data(registry_url, ecr_tag, region)

        logger.info("Making template")
        template_file_path = _make_cloudformation_template(
            project_path,
            encoded_user_data,
            s3_bucket_name,
            sam_template_name,
            elb_name,
            aws_ec2_deployment_config.ami_id,
            aws_ec2_deployment_config.instance_type,
            aws_ec2_deployment_config.autoscale_min_size,
            aws_ec2_deployment_config.autoscale_desired_capacity,
            aws_ec2_deployment_config.autoscale_max_size,
        )
        validate_sam_template(
            sam_template_name, aws_ec2_deployment_config.region, project_path
        )

        logger.info("Building service")
        build_template(
            template_file_path, project_path, aws_ec2_deployment_config.region
        )

        logger.info("Packaging service")
        package_template(s3_bucket_name, project_path, aws_ec2_deployment_config.region)

        logger.info("Deploying service")
        deploy_template(
            deployment_stack_name,
            s3_bucket_name,
            project_path,
            aws_ec2_deployment_config.region,
        )


class AwsEc2DeploymentOperator(DeploymentOperatorBase):
    def add(self, deployment_pb):
        try:
            deployment_spec = deployment_pb.spec
            deployment_spec.aws_ec2_operator_config.region = (
                deployment_spec.aws_ec2_operator_config.region
                or get_default_aws_region()
            )
            if not deployment_spec.aws_ec2_operator_config.region:
                raise InvalidArgument("AWS region is missing")

            ensure_sam_available_or_raise()
            ensure_docker_available_or_raise()

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
            bento_path = bento_pb.bento.uri.uri

            return self._add(deployment_pb, bento_pb, bento_path)
        except BentoMLException as error:
            deployment_pb.state.state = DeploymentState.ERROR
            deployment_pb.state.error_message = f"Error: {str(error)}"
            return ApplyDeploymentResponse(
                status=error.status_proto, deployment=deployment_pb
            )

    def _add(self, deployment_pb, bento_pb, bento_path):
        try:
            if loader._is_remote_path(bento_path):
                with loader._resolve_remote_bundle_path(bento_path) as local_path:
                    return self._add(deployment_pb, bento_pb, local_path)

            deployment_spec = deployment_pb.spec
            aws_ec2_deployment_config = deployment_spec.aws_ec2_operator_config

            user_id = get_aws_user_id()
            artifact_s3_bucket_name = generate_aws_compatible_string(
                "btml-{user_id}-{namespace}".format(
                    user_id=user_id, namespace=deployment_pb.namespace,
                )
            )
            create_s3_bucket_if_not_exists(
                artifact_s3_bucket_name, aws_ec2_deployment_config.region
            )
            deploy_ec2_service(
                deployment_pb,
                deployment_spec,
                bento_path,
                aws_ec2_deployment_config,
                artifact_s3_bucket_name,
                aws_ec2_deployment_config.region,
            )
        except BentoMLException as error:
            if artifact_s3_bucket_name and aws_ec2_deployment_config.region:
                cleanup_s3_bucket_if_exist(
                    artifact_s3_bucket_name, aws_ec2_deployment_config.region
                )
            raise error
        return ApplyDeploymentResponse(status=Status.OK(), deployment=deployment_pb)

    def delete(self, deployment_pb):
        try:
            deployment_spec = deployment_pb.spec
            ec2_deployment_config = deployment_spec.aws_ec2_operator_config
            ec2_deployment_config.region = (
                ec2_deployment_config.region or get_default_aws_region()
            )
            if not ec2_deployment_config.region:
                raise InvalidArgument("AWS region is missing")

            _, deployment_stack_name, repository_name, _ = generate_ec2_resource_names(
                deployment_pb.namespace, deployment_pb.name
            )
            # delete stack
            delete_cloudformation_stack(
                deployment_stack_name, ec2_deployment_config.region
            )

            # delete repo from ecr
            delete_ecr_repository(repository_name, ec2_deployment_config.region)

            # remove bucket
            if deployment_pb.state.info_json:
                deployment_info_json = json.loads(deployment_pb.state.info_json)
                bucket_name = deployment_info_json.get('S3Bucket')
                if bucket_name:
                    cleanup_s3_bucket_if_exist(
                        bucket_name, ec2_deployment_config.region
                    )

            return DeleteDeploymentResponse(status=Status.OK())
        except BentoMLException as error:
            return DeleteDeploymentResponse(status=error.status_proto)

    def update(self, deployment_pb, previous_deployment):
        try:
            ensure_sam_available_or_raise()
            ensure_docker_available_or_raise()
            deployment_spec = deployment_pb.spec
            ec2_deployment_config = deployment_spec.aws_ec2_operator_config
            ec2_deployment_config.region = (
                ec2_deployment_config.region or get_default_aws_region()
            )
            if not ec2_deployment_config.region:
                raise InvalidArgument("AWS region is missing")

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
                deployment_pb,
                previous_deployment,
                bento_pb.bento.uri.uri,
                ec2_deployment_config.region,
            )
        except BentoMLException as error:
            deployment_pb.state.state = DeploymentState.ERROR
            deployment_pb.state.error_message = f"Error: {str(error)}"
            return ApplyDeploymentResponse(
                status=error.status_proto, deployment=deployment_pb
            )

    def _update(self, deployment_pb, previous_deployment_pb, bento_path, region):
        if loader._is_remote_path(bento_path):
            with loader._resolve_remote_bundle_path(bento_path) as local_path:
                return self._update(
                    deployment_pb, previous_deployment_pb, local_path, region
                )

        updated_deployment_spec = deployment_pb.spec
        updated_deployment_config = updated_deployment_spec.aws_ec2_operator_config

        describe_result = self.describe(deployment_pb)
        if describe_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                describe_result.status
            )
            raise YataiDeploymentException(
                f"Failed fetching ec2 deployment current status - "
                f"{error_code}:{error_message}"
            )

        previous_deployment_state = json.loads(describe_result.state.info_json)
        if "S3Bucket" in previous_deployment_state:
            s3_bucket_name = previous_deployment_state.get("S3Bucket")
        else:
            raise BentoMLException(
                "S3 Bucket is missing in the AWS EC2 deployment, please make sure "
                "it exists and try again"
            )

        deploy_ec2_service(
            deployment_pb,
            updated_deployment_spec,
            bento_path,
            updated_deployment_config,
            s3_bucket_name,
            region,
        )

        return ApplyDeploymentResponse(status=Status.OK(), deployment=deployment_pb)

    def describe(self, deployment_pb):
        try:
            deployment_spec = deployment_pb.spec
            ec2_deployment_config = deployment_spec.aws_ec2_operator_config
            ec2_deployment_config.region = (
                ec2_deployment_config.region or get_default_aws_region()
            )
            if not ec2_deployment_config.region:
                raise InvalidArgument("AWS region is missing")

            bento_pb = self.yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            bento_service_metadata = bento_pb.bento.bento_service_metadata
            api_names = [api.name for api in bento_service_metadata.apis]

            _, deployment_stack_name, _, _ = generate_ec2_resource_names(
                deployment_pb.namespace, deployment_pb.name
            )
            try:
                stack_result = describe_cloudformation_stack(
                    ec2_deployment_config.region, deployment_stack_name
                )
                if stack_result.get("Outputs"):
                    outputs = stack_result.get("Outputs")
                else:
                    return DescribeDeploymentResponse(
                        status=Status.ABORTED('"Outputs" field is not present'),
                        state=DeploymentState(
                            state=DeploymentState.ERROR,
                            error_message='"Outputs" field is not present',
                        ),
                    )

                if stack_result["StackStatus"] in FAILED_CLOUDFORMATION_STACK_STATUS:
                    state = DeploymentState(state=DeploymentState.FAILED)
                    return DescribeDeploymentResponse(status=Status.OK(), state=state)

            except Exception as error:  # pylint: disable=broad-except
                state = DeploymentState(
                    state=DeploymentState.ERROR, error_message=str(error)
                )
                return DescribeDeploymentResponse(
                    status=Status.INTERNAL(str(error)), state=state
                )

            info_json = {}
            outputs = {o["OutputKey"]: o["OutputValue"] for o in outputs}
            if "AutoScalingGroup" in outputs:
                info_json["InstanceDetails"] = get_instance_ip_from_scaling_group(
                    [outputs["AutoScalingGroup"]], ec2_deployment_config.region
                )
                info_json["Endpoints"] = get_endpoints_from_instance_address(
                    info_json["InstanceDetails"], api_names
                )
            if "S3Bucket" in outputs:
                info_json["S3Bucket"] = outputs["S3Bucket"]
            if "TargetGroup" in outputs:
                info_json["TargetGroup"] = outputs["TargetGroup"]
            if "Url" in outputs:
                info_json["Url"] = outputs["Url"]

            healthy_target = get_healthy_target(
                outputs["TargetGroup"], ec2_deployment_config.region
            )
            if healthy_target:
                deployment_state = DeploymentState.RUNNING
            else:
                deployment_state = DeploymentState.PENDING
            state = DeploymentState(
                state=deployment_state, info_json=json.dumps(info_json)
            )
            return DescribeDeploymentResponse(status=Status.OK(), state=state)

        except BentoMLException as error:
            return DescribeDeploymentResponse(status=error.status_proto)
