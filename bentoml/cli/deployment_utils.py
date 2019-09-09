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

REQUIRED_DEPLOYMENT_FIELDS = ['name', 'spec']
REQUIRED_DEPLOYMENT_SPEC_FIELDS = ['bento_name', 'bento_version', 'operator']
REQUIRED_SAGEMAKER_FIELDS = ['api_name']


def check_required_fields(yaml_content, required_fields):
    current_keys = list(yaml_content._keys())
    if set(required_fields).issubset(current_keys):
        return
    else:
        missing_keys = [i for i in required_fields if i not in current_keys]
        if len(missing_keys) == 1:
            raise BentoMLException(
                'Required field: {name} is missing'.format(name=missing_keys[0])
            )
        raise BentoMLException(
            'Required fields: {names} are missing'.format(names=','.join(missing_keys))
        )


def deployment_yaml_to_pb(deployment_yaml):
    check_required_fields(deployment_yaml, REQUIRED_DEPLOYMENT_FIELDS)
    spec_yaml = deployment_yaml.get('spec')
    check_required_fields(spec_yaml, REQUIRED_DEPLOYMENT_SPEC_FIELDS)
    platform = spec_yaml.get('operator')
    if platform == 'aws_sagemaker':
        sagemaker_config = spec_yaml.get('sagemaker_operator_config')
        check_required_fields(sagemaker_config, REQUIRED_SAGEMAKER_FIELDS)
        spec = DeploymentSpec(
            sagemaker_operator_config=DeploymentSpec.SageMakerOperatorConfig(
                api_name=sagemaker_config.get('api_name'),
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
                region=lambda_config.get('region', config.get('aws', 'default_region'))
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
        k8s_config = spec_yaml.get('kubernetes_operator_config')
        spec = DeploymentSpec(
            kubernetes_operator_config=DeploymentSpec.KubernetesOperatorConfig(
                kube_namespace=k8s_config.get(
                    'kube_namespace', config.get('kubernetes', 'default_namespace')
                ),
                replicas=k8s_config.get(
                    'replicas', config.get('kubernetes', 'default_replicas')
                ),
                service_name=k8s_config.get(
                    'service_name', deployment_yaml.get('name')
                ),
                service_type=k8s_config.get(
                    'service_type', config.get('kubernetes', 'default_service_type')
                ),
            )
        )
    else:
        raise BentoMLException(
            'Custom deployment is not supported in the current version of BentoML'
        )

    spec.operator = DeploymentOperator.Value(platform.upper())
    spec.bento_name = spec_yaml.get('bento_name')
    spec.bento_version = spec_yaml.get('bento_version')
    return Deployment(
        name=deployment_yaml.get('name'),
        namespace=deployment_yaml.get('namespace'),
        annotations=deployment_yaml.get('annotations'),
        labels=deployment_yaml.get('labels'),
        spec=spec,
    )
