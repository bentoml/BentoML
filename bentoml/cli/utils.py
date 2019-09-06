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

from bentoml.proto.deployment_pb2 import Deployment, DeploymentSpec, DeploymentOperator
from bentoml.exceptions import BentoMLException
from bentoml import config

logger = logging.getLogger(__name__)


def deployment_yaml_to_pb(deployment_yaml):
    try:
        spec_yaml = deployment_yaml['spec']
        platform = spec_yaml['operator']
        if platform == 'aws_sagemaker':
            sagemaker_config = spec_yaml['sagemaker_operator_config']
            spec = DeploymentSpec(
                sagemaker_operator_config=DeploymentSpec.SageMakerOperatorConfig(
                    api_name=sagemaker_config['api_name'],
                    region=sagemaker_config.get(
                        'region', config.get('aws', 'default_region')
                    ),
                    instance_count=sagemaker_config.get(
                        'instance_count', config.getint('sagemaker', 'instance_count')
                    ),
                    instance_type=sagemaker_config.get(
                        'instance_type', config.get('sagemaker', 'instance_type')
                    ),
                )
            )
        elif platform == 'aws_lambda':
            lambda_config = spec_yaml.get('aws_lambda_operator_config', {})
            spec = DeploymentSpec(
                aws_lambda_operator_config=DeploymentSpec.AwsLambdaOperatorConfig(
                    region=lambda_config.get(
                        'region', config.get('aws', 'default_region')
                    )
                )
            )
        elif platform == 'gcp_function':
            gcp_config = spec_yaml.get('gcp_function_operator_config', {})
            spec = DeploymentSpec(
                gcp_function_operator_config=DeploymentSpec.GcpFunctionOperatorConfig(
                    region=gcp_config.get(
                        'region', config.get('google-cloud', 'default_region')
                    )
                )
            )
        elif platform == 'kubernetes':
            k8s_config = spec_yaml['kubernetes_operator_config']
            spec = DeploymentSpec(
                kubernetes_operator_config=DeploymentSpec.KubernetesOperatorConfig(
                    kube_namespace=k8s_config['kube_namespace'],
                    replicas=k8s_config['replicas'],
                    service_name=k8s_config['service_name'],
                    service_type=k8s_config['service_type'],
                )
            )
        else:
            raise BentoMLException(
                'Custom deployment is not supported in the current version of BentoML'
            )

        spec.operator = DeploymentOperator.Value(platform.upper())
        spec.bento_name = spec_yaml['bento_name']
        spec.bento_version = spec_yaml['bento_version']
        return Deployment(
            name=deployment_yaml['name'],
            namespace=deployment_yaml['namespace'],
            annotations=deployment_yaml.get('annotations'),
            labels=deployment_yaml.get('labels'),
            spec=spec,
        )
    except KeyError as e:
        raise BentoMLException('Field {name} is required'.format(name=e.args[0]))
