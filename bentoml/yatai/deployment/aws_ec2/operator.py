"""
# make dockerfile of bentoservice -> done -> automatically generated when service is saved
# build image and tag it properly for ecr repo -> DONE
# push it to ecr -> DONE
#create s3 -> DONE
# make target group -> OPTIONAL
# make launch temaplate which does: -> DONE
#   pull image
#   run it
# create security group which allows traffic on port 5000 -> PENDING

# make autoscaling group for ec2 -> optional
# make template of all above and deploy it -> DONE
"""
import os
import boto3
from pathlib import Path

from bentoml.utils.s3 import is_s3_url
from bentoml.utils.gcs import is_gcs_url
from bentoml.utils.tempdir import TempDirectory
from bentoml.yatai.deployment.aws_lambda.utils import call_sam_command
from bentoml.cli.utils import (
    echo_docker_api_result,
    Spinner,
    get_default_yatai_client,
)
from bentoml.utils import (
    ProtoMessageToDict,
    status_pb_to_error_code_and_message,
)
from bentoml.utils.lazy_loader import LazyLoader
from bentoml.exceptions import BentoMLException

from bentoml.yatai.deployment.operator import DeploymentOperatorBase
from bentoml.yatai.proto.deployment_pb2 import ApplyDeploymentResponse
from bentoml.yatai.status import Status
from bentoml.saved_bundle import (
    load,
    load_bento_service_api,
    load_bento_service_metadata,
)
from bentoml.utils.s3 import create_s3_bucket_if_not_exists
from bentoml.utils.ruamel_yaml import YAML

yatai_proto = LazyLoader("yatai_proto", globals(), "bentoml.yatai.proto")

def to_valid_docker_image_name(name):
    # https://docs.docker.com/engine/reference/commandline/tag/#extended-description
    return name.lower().strip("._-")

def to_valid_docker_image_version(version):
    # https://docs.docker.com/engine/reference/commandline/tag/#extended-description
    return version.encode("ascii", errors="ignore").decode().lstrip(".-")[:128]


def resolve_bundle_path(bento, pip_installed_bundle_path):
    if pip_installed_bundle_path:
        assert (
            bento is None
        ), "pip installed BentoService commands should not have Bento argument"
        return pip_installed_bundle_path

    if os.path.isdir(bento) or is_s3_url(bento) or is_gcs_url(bento):
        # saved_bundle already support loading local, s3 path and gcs path
        return bento

    elif ":" in bento:
        # assuming passing in BentoService in the form of Name:Version tag
        yatai_client = get_default_yatai_client()
        name, version = bento.split(":")
        get_bento_result = yatai_client.repository.get(name, version)
        if get_bento_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_bento_result.status
            )
            raise BentoMLException(
                f"BentoService {name}:{version} not found - "
                f"{error_code}:{error_message}"
            )
        if get_bento_result.bento.uri.s3_presigned_url:
            # Use s3 presigned URL for downloading the repository if it is presented
            return get_bento_result.bento.uri.s3_presigned_url
        if get_bento_result.bento.uri.gcs_presigned_url:
            return get_bento_result.bento.uri.gcs_presigned_url
        else:
            return get_bento_result.bento.uri.uri
    else:
        raise BentoMLException(
            f'BentoService "{bento}" not found - either specify the file path of '
            f"the BentoService saved bundle, or the BentoService id in the form of "
            f'"name:version"'
        )

def _containerize_service(bento, docker_api, tag=None, build_arg = None, username = None, password = None):
    saved_bundle_path = resolve_bundle_path(bento, None)

    print(f"Found Bento: {saved_bundle_path}")

    bento_metadata = load_bento_service_metadata(saved_bundle_path)
    name = to_valid_docker_image_name(bento_metadata.name)
    version = to_valid_docker_image_version(bento_metadata.version)

    if not tag:
        print(
            "Tag not specified, using tag parsed from "
            f"BentoService: '{name}:{version}'"
        )
        tag = f"{name}:{version}"
    if ":" not in tag:
        print(
            "Image version not specified, using version parsed "
            f"from BentoService: {version}",
        )
        tag = f"{tag}:{version}"

    docker_build_args = {}
    if build_arg:
        for arg in build_arg:
            key, value = arg.split("=")
            docker_build_args[key] = value

    import docker

    #docker_api = docker.APIClient()
    try:
        with Spinner(f"Building Docker image {tag} from {bento} \n"):
            for line in echo_docker_api_result(
                docker_api.build(
                    path=saved_bundle_path,
                    tag=tag,
                    decode=True,
                    buildargs=docker_build_args,
                )
            ):
                print(line)
    except docker.errors.APIError as error:
        raise BentoMLException(f"Could not build Docker image: {error}")


    print("pushing image")
    auth_config_payload = (
        {"username": username, "password": password}
        if username or password
        else None
    )

    try:
        with Spinner(f"Pushing docker image to {tag}\n"):
            for line in echo_docker_api_result(
                docker_api.push(
                    repository=tag,
                    stream=True,
                    decode=True,
                    auth_config=auth_config_payload,
                )
            ):
                print(line)
        print(
            f"Pushed {tag} to {name}",
        )
    except (docker.errors.APIError, BentoMLException) as error:
        raise BentoMLException (f"Could not push Docker image: {error}")

def _create_ecr_repo(repo_name):
    ecr_client = boto3.client("ecr")
    repository = ecr_client.create_repository(repositoryName = repo_name, imageScanningConfiguration = {"scanOnPush" : False})
    registry_id = repository["repository"]["registryId"]
    return registry_id

def _get_ecr_password(registry_id):
    ecr_client = boto3.client("ecr")
    token_data = ecr_client.get_authorization_token(registryIds = [registry_id])
    token = token_data["authorizationData"][0]["authorizationToken"]
    registry_endpoint = token_data["authorizationData"][0]["proxyEndpoint"]
    return token, registry_endpoint

def _get_creds_from_token(token):
    import base64
    cred_string = base64.b64decode(token).decode("ascii")
    username, password = str(cred_string).split(":")
    return username, password

def _login_docker(username, password, registry_url):
    import docker

    docker_api = docker.APIClient()
    docker_api.login(username = username, password = password, registry = registry_url)
    return docker_api

def _make_user_data(username, password, registry, tag):
    #input: docker usernaame, password, registry
    #bento image name
    #return base64 encoded user data
    import base64

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
""".format(username, password, registry, tag)

    print("user data is \n", base_format)
    encoded = base64.b64encode(base_format.encode("ascii")).decode("ascii")
    print("encoded data is \n", encoded)
    return encoded

def _make_cloudformation_template(project_dir, user_data):
    template_file_path = os.path.join(project_dir, "template.yml")
    yaml = YAML()
    sam_config = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Transform": "AWS::Serverless-2016-10-31",
        "Description": "BentoML load balanced template template",
        "Parameters" : {
            "AmazonLinux2LatestAmiId":{
                "Type": "AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>",
                "Default": "/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2"
            },
        }
    }
    yaml.dump(sam_config, Path(template_file_path))
    #NOTE: REMOVE SSH ACCESS TO INSTANCE FROM BELOW TEMPLATE

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
            """.format(user_data)
        )
    return template_file_path


def deploy_template(project_directory, template_file_path, s3_bucket_name, region, stack_name):
    #build -> sam build -t "$2".yaml

    status_code, stdout, stderr = call_sam_command(["build", "-t", template_file_path], project_directory, region)
    print("build ", status_code, stdout, stderr)


    #package -> sam package --output-template-file packaged.yaml --s3-bucket "$1"
    status_code, stdout, stderr = call_sam_command(["package", "--output-template-file", "packaged.yaml", "--s3-bucket", s3_bucket_name], project_directory, region)
    print("package ", status_code, stdout, stderr)

    #deploy -> sam deploy --template-file packaged.yaml --stack-name "$3" --capabilities CAPABILITY_IAM --s3-bucket "$1"
    status_code, stdout, stderr = call_sam_command(["deploy", "--template-file", "packaged.yaml", "--stack-name", stack_name, "--capabilities", "CAPABILITY_IAM", "--s3-bucket", s3_bucket_name], project_directory, region)
    print("deploy ", status_code, stdout, stderr)

class AwsEc2DeploymentOperator(DeploymentOperatorBase):
    def add(self, deployment_pb):
        import uuid
        REPO_NAME = str(uuid.uuid4())
        REGION = "us-east-1"
        S3_BUCKET_NAME = "bento-iris-classifier-1234"
        STACK_NAME = "bento-stack-auto-2"
        TAG = "752014255238.dkr.ecr.ap-south-1.amazonaws.com/bento-iris:latest"

        with TempDirectory() as PROJECT_PATH:
            print("deployment_pb is ", deployment_pb)
            bento = f"{deployment_pb.spec.bento_name}:latest"

            
            registry_id = _create_ecr_repo(REPO_NAME)
            registry_token, registry_url = _get_ecr_password(registry_id)
            registry_username, registry_password = _get_creds_from_token(registry_token)

            docker_api = _login_docker(registry_username, registry_password, registry_url)
            _containerize_service(bento, docker_api, tag = TAG, username=registry_username, password=registry_password)
            
            create_s3_bucket_if_not_exists(
                    S3_BUCKET_NAME, REGION
                )

            encoded_user_data = _make_user_data(registry_username, registry_password, registry_url, TAG)
            
            template_file_path = _make_cloudformation_template(PROJECT_PATH, encoded_user_data)

            #validate_template_file() TODO

            deploy_template(PROJECT_PATH, template_file_path, S3_BUCKET_NAME, REGION, STACK_NAME)


        return ApplyDeploymentResponse(status=Status.OK(), deployment=deployment_pb)
    def update(self, deployment_pb):
        pass
    def delete(self, deployment_pb):
        pass
    def describe(self, deployment_pb):
        pass
