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

from bentoml.proto.deployment_pb2 import Deployment, DeploymentSpec
from bentoml.exceptions import BentoMLException

logger = logging.getLogger(__name__)


def deployment_yaml_to_pb(deployment_yaml):
    deployment_pb = Deployment()

    if deployment_yaml.get('name') is not None:
        deployment_pb.name = deployment_yaml.get('name')
    if deployment_yaml.get('namespace') is not None:
        deployment_pb.namespace = deployment_yaml.get('namespace')
    if deployment_yaml.get('labels') is not None:
        deployment_pb.labels.update(dict(deployment_yaml.get('labels')))
    if deployment_yaml.get('annotations') is not None:
        deployment_pb.annotations.update(dict(deployment_yaml.get('annotations')))

    spec_yaml = deployment_yaml.get('spec')
    platform = spec_yaml.get('operator')
    if platform is not None:
        deployment_pb.spec.operator = DeploymentSpec.DeploymentOperator.Value(
            platform.replace('-', '_').upper()
        )
    if spec_yaml.get('bento_name'):
        deployment_pb.spec.bento_name = spec_yaml.get('bento_name')
    if spec_yaml.get('bento_version'):
        deployment_pb.spec.bento_version = spec_yaml.get('bento_version')

    if platform == 'aws_sagemaker':
        sagemaker_config = spec_yaml.get('sagemaker_operator_config')
        sagemaker_operator_config_pb = deployment_pb.spec.sagemaker_operator_config
        if sagemaker_config.get('api_name'):
            sagemaker_operator_config_pb.api_name = sagemaker_config.get('api_name')
        if sagemaker_config.get('region'):
            sagemaker_operator_config_pb.region = sagemaker_config.get('region')
        if sagemaker_config.get('instance_count'):
            sagemaker_operator_config_pb.instance_count = sagemaker_config.get(
                'instance_count'
            )
        if sagemaker_config.get('instance_type'):
            sagemaker_operator_config_pb.instance_type = sagemaker_config.get(
                'instance_type'
            )
    elif platform == 'aws_lambda':
        lambda_config = spec_yaml.get('aws_lambda_operator_config')
        if lambda_config.get('region'):
            deployment_pb.spec.aws_lambda_config.region = lambda_config.get('region')
    elif platform == 'gcp_function':
        gcp_config = spec_yaml.get('gcp_function_operator_config')
        if gcp_config.get('region'):
            deployment_pb.spec.gcp_function_operator_config.region = gcp_config.get(
                'region'
            )
    elif platform == 'kubernetes':
        k8s_config = spec_yaml.get('kubernetes_operator_config')
        k8s_operator_config_pb = deployment_pb.spec.kubernetes_operator_config

        if k8s_config.get('kube_namespace'):
            k8s_operator_config_pb.kube_namespace = k8s_config.get('kube_namespace')
        if k8s_config.get('replicas'):
            k8s_operator_config_pb.replicas = k8s_config.get('replicas')
        if k8s_config.get('service_name'):
            k8s_operator_config_pb.service_name = k8s_config.get('service_name')
        if k8s_config.get('service_type'):
            k8s_operator_config_pb.service_type = k8s_config.get('service_type')
    else:
        raise BentoMLException(
            'Custom deployment is not supported in the current version of BentoML'
        )

    return deployment_pb
