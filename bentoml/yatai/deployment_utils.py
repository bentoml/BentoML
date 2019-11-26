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


import logging

from ruamel.yaml import YAML

from bentoml.proto.deployment_pb2 import Deployment, DeploymentSpec
from bentoml.exceptions import BentoMLException, BentoMLDeploymentException

logger = logging.getLogger(__name__)


def deployment_dict_to_pb(deployment_dict):
    deployment_pb = Deployment()
    if deployment_dict.get('name') is not None:
        deployment_pb.name = deployment_dict.get('name')
    if deployment_dict.get('namespace') is not None:
        deployment_pb.namespace = deployment_dict.get('namespace')
    if deployment_dict.get('labels') is not None:
        deployment_pb.labels.update(deployment_dict.get('labels'))
    if deployment_dict.get('annotations') is not None:
        deployment_pb.annotations.update(deployment_dict.get('annotations'))

    if deployment_dict.get('spec'):
        spec_dict = deployment_dict.get('spec')
    else:
        raise BentoMLDeploymentException('"spec" is required field for deployment')
    platform = spec_dict.get('operator')
    if platform is not None:
        # converting platform parameter to DeploymentOperator name in proto
        # e.g. 'aws-lambda' to 'AWS_LAMBDA'
        deployment_pb.spec.operator = DeploymentSpec.DeploymentOperator.Value(
            platform.replace('-', '_').upper()
        )

    if spec_dict.get('bento_name'):
        deployment_pb.spec.bento_name = spec_dict.get('bento_name')
    if spec_dict.get('bento_version'):
        deployment_pb.spec.bento_version = spec_dict.get('bento_version')

    if deployment_pb.spec.operator == DeploymentSpec.AWS_SAGEMAKER:
        sagemaker_config = spec_dict.get('sagemaker_operator_config', {})
        sagemaker_operator_config_pb = deployment_pb.spec.sagemaker_operator_config
        if sagemaker_config.get('api_name'):
            sagemaker_operator_config_pb.api_name = sagemaker_config.get('api_name')
        if sagemaker_config.get('region'):
            sagemaker_operator_config_pb.region = sagemaker_config.get('region')
        if sagemaker_config.get('instance_count'):
            sagemaker_operator_config_pb.instance_count = int(
                sagemaker_config.get('instance_count')
            )
        if sagemaker_config.get('instance_type'):
            sagemaker_operator_config_pb.instance_type = sagemaker_config.get(
                'instance_type'
            )
    elif deployment_pb.spec.operator == DeploymentSpec.AWS_LAMBDA:
        lambda_conf = spec_dict.get('aws_lambda_operator_config', {})
        for field in ['region', 'api_name', 'memory_size', 'timeout']:
            if lambda_conf.get(field):
                deployment_pb.spec.aws_lambda_operator_config.__setattr__(
                    field, lambda_conf.get(field)
                )
    elif deployment_pb.spec.operator == DeploymentSpec.GCP_FUNCTION:
        gcp_config = spec_dict.get('gcp_function_operator_config', {})
        for field in ['region', 'api_name']:
            if gcp_config.get(field):
                deployment_pb.spec.gcp_function_operator_config.__setattr__(
                    field, gcp_config.get(field)
                )
    elif deployment_pb.spec.operator == DeploymentSpec.KUBERNETES:
        k8s_config = spec_dict.get('kubernetes_operator_config', {})

        for field in ['kube_namespace', 'replicas', 'service_name', 'service_type']:
            if k8s_config.get(field):
                deployment_pb.spec.kubernetes_operator_config.__setattr__(
                    field, k8s_config.get(field)
                )
    else:
        raise BentoMLException(
            'Platform "{}" is not supported in the current version of '
            'BentoML'.format(platform)
        )

    return deployment_pb


def deployment_yaml_string_to_pb(deployment_yaml_string):
    yaml = YAML()
    deployment_yaml = yaml.load(deployment_yaml_string)
    return deployment_dict_to_pb(deployment_yaml)
