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
#from botocore.exceptions import RepositoryAlreadyExistsException
from pathlib import Path
from uuid import uuid4

from bentoml.utils.tempdir import TempDirectory

from bentoml.utils.lazy_loader import LazyLoader
from bentoml.saved_bundle import loader
from bentoml.utils import (
    ProtoMessageToDict,
    status_pb_to_error_code_and_message,
    resolve_bundle_path
)
from bentoml.yatai.deployment.operator import DeploymentOperatorBase
from bentoml.yatai.proto.deployment_pb2 import ApplyDeploymentResponse
from bentoml.yatai.status import Status
from bentoml.saved_bundle import (
    load_bento_service_metadata
)
from bentoml.utils.s3 import create_s3_bucket_if_not_exists
from bentoml.utils.ruamel_yaml import YAML
from bentoml.yatai.deployment.utils import ensure_docker_available_or_raise
from bentoml.yatai.deployment.aws_utils import (
    generate_aws_compatible_string,
    get_default_aws_region,
    ensure_sam_available_or_raise,
    call_sam_command,
    validate_sam_template
)
from bentoml.utils.docker_utils import (
    to_valid_docker_image_name,
    to_valid_docker_image_version,
    validate_tag,
    containerize_bento_service
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
        repository = ecr_client.create_repository(repositoryName = repo_name, imageScanningConfiguration = {"scanOnPush" : False})
        registry_id = repository["repository"]["registryId"]
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        all_repositories = ecr_client.describe_repositories(repositoryNames=[repo_name])
        registry_id = all_repositories['repositories'][0]["registryId"]
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
    encoded = base64.b64encode(base_format.encode("ascii")).decode("ascii")
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
        print("deployment pb is \n", deployment_pb)

        deployment_spec = deployment_pb.spec
        deployment_spec.aws_ec2_operator_config.region = (
            deployment_spec.aws_ec2_operator_config.region or get_default_aws_region()
        )
        if not deployment_spec.aws_ec2_operator_config:
            raise InvalidArgument("AWS region is missing")
        #REGION = "us-east-1"

        ensure_sam_available_or_raise()
        ensure_docker_available_or_raise()

        bento_pb = self.yatai_service.GetBento(
            GetBentoRequest(
                bento_name=deployment_spec.bento_name,
                bento_version = deployment_spec.bento_version
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
        #bento_service_metadata = bento_pb.bento.bento_service_metadata

        artifact_s3_bucket_name = generate_aws_compatible_string(
            "btml-{namespace}-{name}-{random_string}".format(
                namespace=deployment_pb.namespace,
                name=deployment_pb.name,
                random_string=uuid4().hex[:6].lower()
            )
        )
        #S3_BUCKET_NAME = "bento-iris-classifier-1234"
        create_s3_bucket_if_not_exists(
            artifact_s3_bucket_name, aws_ec2_deployment_config.region
        )

        deployment_stack_name = generate_aws_compatible_string(
            "btml-stack-{namespace}-{name}-{random_string}".format(
                namespace=deployment_pb.namespace,
                name=deployment_pb.name,
                random_string=uuid4().hex[:6].lower()
            )
        )

        repo_name = generate_aws_compatible_string(
            "btml-repo-{namespace}-{name}-{random_string}".format(
                namespace=deployment_pb.namespace,
                name=deployment_pb.name,
                random_string=uuid4().hex[:6].lower()
            )
        )
        repo_name="bento-iris" #NOTE: DELETE THIS

        with TempDirectory() as project_path:
            registry_id = _create_ecr_repo(repo_name)
            registry_token, registry_url = _get_ecr_password(registry_id)
            registry_username, registry_password = _get_creds_from_token(registry_token)
            
            #docker_api = _login_docker(registry_username, registry_password, registry_url)
            #752014255238.dkr.ecr.ap-south-1.amazonaws.com/btml-repo-dev-deploy-76-cdb0ae:latest
            registry_domain = registry_url.replace("https://", "")
            tag = f"{registry_domain}/{repo_name}"

            #containerize_bento_service(bento, True, tag = tag, 
            #    build_arg = {}, username=registry_username, 
            #    password=registry_password, pip_installed_bundle_path=None)
            
            encoded_user_data = _make_user_data(registry_username, registry_password, registry_url, tag)
            
            template_file_path = _make_cloudformation_template(project_path, encoded_user_data)

            validate_sam_template("template.yml", aws_ec2_deployment_config.region, project_path)

            """
            deploy_template(project_path, template_file_path, 
                            artifact_s3_bucket_name, aws_ec2_deployment_config.region, 
                            deployment_stack_name)
            """
        return ApplyDeploymentResponse(status=Status.OK(), deployment=deployment_pb)
        
    def update(self, deployment_pb):
        pass

    def delete(self, deployment_pb):
        #delete stack
        deployment_spec = deployment_pb


        #delete repo from ecr


    def describe(self, deployment_pb):
        pass
