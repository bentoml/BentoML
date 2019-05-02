# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import base64
import subprocess
import logging
import re
from six.moves.urllib.parse import urlparse

import boto3
import docker

from bentoml.archive import load
from bentoml.deployment.utils import generate_bentoml_deployment_snapshot_path

logger = logging.getLogger(__name__)

DEFAULT_REGION = 'us-west-2'
DEFAULT_INSTANCE_TYPE = 'ml.m4.xlarge'
DEFAULT_INITIAL_INSTANCE_COUNT = 1


def strip_scheme(url):
    """ Stripe url's schema
    e.g.   http://some.url/path -> some.url/path
    :param url: String
    :return: String
    """
    parsed = urlparse(url)
    scheme = "%s://" % parsed.scheme
    return parsed.geturl().replace(scheme, '', 1)


def generate_aws_compatible_string(item):
    pattern = re.compile('[^a-zA-Z0-9-]|_')
    return re.sub(pattern, '-', item)


def get_arn_role_from_current_user():
    sts_client = boto3.client("sts")
    identity = sts_client.get_caller_identity()
    sts_arn = identity["Arn"]
    sts_arn_list = sts_arn.split(':')
    type_role = sts_arn_list[-1].split('/')
    iam_client = boto3.client("iam")
    if type_role[0] == 'user':
        role_list = iam_client.list_roles()
        arn = None
        for role in role_list['Roles']:
            policy_document = role['AssumeRolePolicyDocument']
            statement = policy_document['Statement'][0]
            if statement['Effect'] == 'Allow' and statement['Principal'][
                    'Service'] == 'sagemaker.amazonaws.com':
                arn = role['Arn']
        if arn is None:
            raise ValueError(
                "Can't find proper Arn role for Sagemaker, please create one and try again")
        return arn
    elif type_role[0] == 'role':
        role_response = iam_client.get_role(RoleName=type_role[1])
        return role_response["Role"]["Arn"]


def create_push_image_to_ecr(bento_service, snapshot_path):
    """Create BentoService sagemaker image and push to AWS ECR

    :param bento_service: Bento Service
    :param snapshot_path: Path
    :return: ecr_tag: String
    """

    # Example:
    # https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/scikit_bring_your_own/container/build_and_push.sh
    # 1. get aws account info and login ecr
    # 2. create ecr repository, if not exist
    # 3. build tag and push docker image

    ecr_client = boto3.client('ecr')
    token = ecr_client.get_authorization_token()
    logger.info('Getting docker login info from AWS')
    username, password = base64.b64decode(
        token['authorizationData'][0]['authorizationToken']).decode('utf-8').split(":")
    registry_url = token['authorizationData'][0]['proxyEndpoint']

    # https://github.com/docker/docker-py/issues/2256
    subprocess.call(['docker', 'login', '-u', username, '-p', password, registry_url])
    docker_client = docker.from_env()
    docker_client.login(username, password, email='', registry=registry_url)

    image_name = bento_service.name.lower() + '-sagemaker'
    ecr_tag = strip_scheme('{registry_url}/{image_name}:{version}'.format(
        registry_url=registry_url, image_name=image_name, version=bento_service.version))

    try:
        ecr_client.describe_repositories(repositoryNames=[image_name])['repositories']
    except ecr_client.exceptions.RepositoryNotFoundException:
        ecr_client.create_repository(repositoryName=image_name)

    print('Building docker image: {}'.format(image_name))
    built_image = docker_client.images.build(path=snapshot_path, tag=image_name)
    built_image[0].tag(ecr_tag)
    print('Pushing image to AWS ECR at {}'.format(ecr_tag))
    docker_client.api.push(repository=ecr_tag)
    print('Finished pushing image: {}'.format(ecr_tag))
    return ecr_tag


def deploy_with_sagemaker(archive_path, additional_info):
    """Deploy BentoML service to AWS Sagemaker.
    Your AWS credential must have the correct permissions for sagemaker and ECR

    :param archive_path: Path
    :param additional_info: Dict
    :return:
    """

    # 1. Create docker image and push to ECR
    # 2. Create sagemaker model base on the ECR image
    # 2. Create sagemaker endpoint configuration base on sagemaker model
    # 3. Create sagemaker endpoint base on sagemaker endpoint configuration
    bento_service = load(archive_path)
    snapshot_path = generate_bentoml_deployment_snapshot_path(bento_service.name, 'aws-sagemaker')
    shutil.copytree(archive_path, snapshot_path)
    region = additional_info.get('region', DEFAULT_REGION)
    execution_role_arn = get_arn_role_from_current_user()
    ecr_image_path = create_push_image_to_ecr(bento_service, snapshot_path)
    sagemaker_model_name = generate_aws_compatible_string('bentoml-' + bento_service.name + '-' +
                                                          bento_service.version)
    sagemaker_client = boto3.client('sagemaker', region_name=region)

    sagemaker_model_info = {
        "ModelName": sagemaker_model_name,
        "PrimaryContainer": {
            "ContainerHostname": sagemaker_model_name,
            "Image": ecr_image_path,
        },
        "ExecutionRoleArn": execution_role_arn,
    }
    print('Creating sagemaker model {}'.format(sagemaker_model_name))
    model_response = sagemaker_client.create_model(**sagemaker_model_info)
    logger.info(model_response)

    endpoint_config_name = generate_aws_compatible_string(bento_service.name + '-' +
                                                          bento_service.version + '-configuration')
    production_variants = [{
        "VariantName": bento_service.name,
        "ModelName": sagemaker_model_name,
        "InitialInstanceCount": DEFAULT_INITIAL_INSTANCE_COUNT,
        "InstanceType": DEFAULT_INSTANCE_TYPE,
    }]
    print('Creating sagemaker endpoint {} configuration'.format(endpoint_config_name))
    create_endpoint_config_response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name, ProductionVariants=production_variants)
    logger.info(create_endpoint_config_response)
    print('Creating sagemaker endpoint {}'.format(bento_service.name))
    create_endpoint_response = sagemaker_client.create_endpoint(
        EndpointName=bento_service.name, EndpointConfigName=endpoint_config_name)
    logger.info(create_endpoint_response)

    endpoint_status = sagemaker_client.describe_endpoint(EndpointName=bento_service.name)
    logger.info(endpoint_status)
    print(endpoint_status)
    return
