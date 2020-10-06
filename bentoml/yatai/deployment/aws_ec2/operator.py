import os
import boto3
from pathlib import Path
from uuid import uuid4
import docker
import base64

from bentoml.utils.tempdir import TempDirectory

from bentoml.utils.lazy_loader import LazyLoader
from bentoml.saved_bundle import loader
from bentoml.yatai.deployment.operator import DeploymentOperatorBase
from bentoml.yatai.proto.deployment_pb2 import (
    ApplyDeploymentResponse,
    DeleteDeploymentResponse,
)
from bentoml.yatai.status import Status

from bentoml.utils.s3 import create_s3_bucket_if_not_exists
from bentoml.utils.ruamel_yaml import YAML
from bentoml.yatai.deployment.utils import ensure_docker_available_or_raise
from bentoml.yatai.deployment.aws_utils import (
    generate_aws_compatible_string,
    get_default_aws_region,
    ensure_sam_available_or_raise,
    call_sam_command,
    validate_sam_template,
)
from bentoml.utils.docker_utils import (
    to_valid_docker_image_name,
    to_valid_docker_image_version,
    validate_tag,
    containerize_bento_service,
)
from bentoml.exceptions import (
    BentoMLException,
    InvalidArgument,
    YataiDeploymentException,
)
from bentoml.yatai.proto.repository_pb2 import GetBentoRequest, BentoUri

yatai_proto = LazyLoader("yatai_proto", globals(), "bentoml.yatai.proto")


def _create_ecr_repo(repo_name):
    try:
        ecr_client = boto3.client("ecr")
        repository = ecr_client.create_repository(
            repositoryName=repo_name, imageScanningConfiguration={"scanOnPush": False}
        )
        registry_id = repository["repository"]["registryId"]
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        all_repositories = ecr_client.describe_repositories(repositoryNames=[repo_name])
        registry_id = all_repositories["repositories"][0]["registryId"]
    return registry_id


def _get_ecr_password(registry_id):
    ecr_client = boto3.client("ecr")
    token_data = ecr_client.get_authorization_token(registryIds=[registry_id])
    token = token_data["authorizationData"][0]["authorizationToken"]
    registry_endpoint = token_data["authorizationData"][0]["proxyEndpoint"]
    return token, registry_endpoint


def _get_creds_from_token(token):
    cred_string = base64.b64decode(token).decode("ascii")
    username, password = str(cred_string).split(":")
    return username, password


def _login_docker(username, password, registry_url):
    docker_api = docker.APIClient()
    docker_api.login(username=username, password=password, registry=registry_url)
    return docker_api


def _make_user_data(username, password, registry, tag):
    base_format = """MIME-Version: 1.0
Content-Type: multipart/mixed; boundary=\"==MYBOUNDARY==\"

--==MYBOUNDARY==
Content-Type: text/cloud-config; charset=\"us-ascii\"

runcmd:

- sudo yum update -y
- sudo amazon-linux-extras install docker -y
- sudo service docker start
- sudo usermod -a -G docker ec2-user
- curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip'
- unzip awscliv2.zip
- sudo ./aws/install
- ln -s /usr/bin/aws aws
- docker login --username {0} --password {1} {2}
- sudo docker pull {3}
- sudo docker run -p 5000:5000 {3}

--==MYBOUNDARY==--
""".format(
        username, password, registry, tag
    )
    encoded = base64.b64encode(base_format.encode("ascii")).decode("ascii")
    return encoded


def _make_cloudformation_template(project_dir, user_data):
    """
    TODO: Template should return ec2 public address
    NOTE: SSH ACCESS TO INSTANCE MAY NOT BE REQUIRED
    """
    template_file_path = os.path.join(project_dir, "template.yml")
    yaml = YAML()
    sam_config = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Transform": "AWS::Serverless-2016-10-31",
        "Description": "BentoML load balanced template template",
        "Parameters": {
            "AmazonLinux2LatestAmiId": {
                "Type": "AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>",
                "Default": "/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2",
            },
        },
    }
    yaml.dump(sam_config, Path(template_file_path))

    with open(template_file_path, "a") as f:
        f.write(
            """\
Resources:
    SecurityGroupResource:
        Type: AWS::EC2::SecurityGroup
        Properties:
            GroupDescription: "security group for bentoservice"
            SecurityGroupIngress:
                -
                    IpProtocol: tcp
                    CidrIp: 0.0.0.0/0
                    FromPort: 5000
                    ToPort: 5000
                -
                    IpProtocol: tcp
                    CidrIp: 0.0.0.0/0
                    FromPort: 22
                    ToPort: 22

    LaunchTemplateResource:
        Type: AWS::EC2::LaunchTemplate
        Properties:
            LaunchTemplateName: template-1
            #Key and security gorups remainign for logging in
            LaunchTemplateData:
                ImageId: !Ref AmazonLinux2LatestAmiId
                InstanceType: t2.micro
                UserData: "{0}"
                SecurityGroupIds:
                - !GetAtt SecurityGroupResource.GroupId
    
    AutoScalingGroup:
        Type: AWS::AutoScaling::AutoScalingGroup
        Properties:
            MinSize: "0"
            MaxSize: "1"
            DesiredCapacity: "1"
            AvailabilityZones: !GetAZs
            LaunchTemplate: 
                LaunchTemplateId: !Ref LaunchTemplateResource
                Version: !GetAtt LaunchTemplateResource.LatestVersionNumber
            """.format(
                user_data
            )
        )
    return template_file_path


def deploy_template(
    project_directory, template_file_path, s3_bucket_name, region, stack_name
):
    """
    TODO: make separate function for package,build,deploy
    """
    status_code, stdout, stderr = call_sam_command(
        ["build", "-t", template_file_path], project_directory, region
    )

    status_code, stdout, stderr = call_sam_command(
        [
            "package",
            "--output-template-file",
            "packaged.yaml",
            "--s3-bucket",
            s3_bucket_name,
        ],
        project_directory,
        region,
    )

    status_code, stdout, stderr = call_sam_command(
        [
            "deploy",
            "--template-file",
            "packaged.yaml",
            "--stack-name",
            stack_name,
            "--capabilities",
            "CAPABILITY_IAM",
            "--s3-bucket",
            s3_bucket_name,
        ],
        project_directory,
        region,
    )


class AwsEc2DeploymentOperator(DeploymentOperatorBase):
    def add(self, deployment_pb):

        deployment_spec = deployment_pb.spec
        deployment_spec.aws_ec2_operator_config.region = (
            deployment_spec.aws_ec2_operator_config.region or get_default_aws_region()
        )
        if not deployment_spec.aws_ec2_operator_config:
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

    def _add(self, deployment_pb, bento_pb, bento_path):
        if loader._is_remote_path(bento_path):
            with loader._resolve_remote_bundle_path(bento_path) as local_path:
                return self._add(deployment_pb, bento_pb, local_path)

        deployment_spec = deployment_pb.spec
        aws_ec2_deployment_config = deployment_spec.aws_ec2_operator_config

        artifact_s3_bucket_name = generate_aws_compatible_string(
            "btml-{namespace}-{name}-{random_string}".format(
                namespace=deployment_pb.namespace,
                name=deployment_pb.name,
                random_string=uuid4().hex[:6].lower(),
            )
        )
        create_s3_bucket_if_not_exists(
            artifact_s3_bucket_name, aws_ec2_deployment_config.region
        )

        deployment_stack_name = generate_aws_compatible_string(
            "btml-stack-{namespace}-{name}".format(
                namespace=deployment_pb.namespace, name=deployment_pb.name
            )
        )

        repo_name = generate_aws_compatible_string(
            "btml-repo-{namespace}-{name}".format(
                namespace=deployment_pb.namespace, name=deployment_pb.name
            )
        )

        with TempDirectory() as project_path:
            registry_id = _create_ecr_repo(repo_name)
            registry_token, registry_url = _get_ecr_password(registry_id)
            registry_username, registry_password = _get_creds_from_token(registry_token)

            bento = f"{deployment_pb.spec.bento_name}:latest"
            registry_domain = registry_url.replace("https://", "")
            tag = f"{registry_domain}/{repo_name}"

            containerize_bento_service(
                bento,
                True,
                tag=tag,
                build_arg={},
                username=registry_username,
                password=registry_password,
                pip_installed_bundle_path=None,
            )

            encoded_user_data = _make_user_data(
                registry_username, registry_password, registry_url, tag
            )

            template_file_path = _make_cloudformation_template(
                project_path, encoded_user_data
            )

            validate_sam_template(
                "template.yml", aws_ec2_deployment_config.region, project_path
            )

            deploy_template(
                project_path,
                template_file_path,
                artifact_s3_bucket_name,
                aws_ec2_deployment_config.region,
                deployment_stack_name,
            )

        return ApplyDeploymentResponse(status=Status.OK(), deployment=deployment_pb)

    def delete(self, deployment_pb):
        deployment_spec = deployment_pb.spec
        ec2_deployment_config = deployment_spec.aws_ec2_operator_config
        ec2_deployment_config.region = (
            ec2_deployment_config.region or get_default_aws_region()
        )
        if not ec2_deployment_config.region:
            raise InvalidArgument("AWS region is missing")

        # delete stack
        deployment_spec = deployment_pb
        deployment_stack_name = generate_aws_compatible_string(
            "btml-stack-{namespace}-{name}".format(
                namespace=deployment_pb.namespace, name=deployment_pb.name
            )
        )

        cf_client = boto3.client("cloudformation", ec2_deployment_config.region)
        cf_client.delete_stack(StackName=deployment_stack_name)

        # delete repo from ecr
        repository_name = generate_aws_compatible_string(
            "btml-repo-{namespace}-{name}".format(
                namespace=deployment_pb.namespace, name=deployment_pb.name
            )
        )
        ecr_client = boto3.client("ecr", ec2_deployment_config.region)
        ecr_client.delete_repository(repositoryName=repository_name)
        return DeleteDeploymentResponse(status=Status.OK())

    def update(self, deployment_pb, previous_deployment):
        pass

    def describe(self, deployment_pb):
        pass
