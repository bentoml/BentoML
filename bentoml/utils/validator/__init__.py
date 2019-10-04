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

from cerberus import Validator

from bentoml.utils import ProtoMessageToDict
from bentoml.proto.deployment_pb2 import DeploymentSpec

deployment_schema = {
    'name': {'type': 'string', 'required': True, 'minlength': 4},
    'namespace': {'type': 'string', 'required': True, 'minlength': 4},
    'labels': {'type': 'dict', 'allow_unknown': True},
    'annotations': {'type': 'dict', 'allow_unknown': True},
    'spec': {
        'type': 'dict',
        'required': True,
        'schema': {
            'operator': {
                'type': 'string',
                'required': True,
                'allowed': DeploymentSpec.DeploymentOperator.keys(),
            },
            'bento_name': {'type': 'string', 'required': True},
            'bento_version': {'type': 'string', 'required': True},
            'custom_operator_config': {
                'type': 'dict',
                'schema': {
                    'name': {'type': 'string'},
                    'config': {'type': 'dict', 'allow_unknown': True},
                },
            },
            'sagemaker_operator_config': {
                'type': 'dict',
                'schema': {
                    'region': {'type': 'string'},
                    'api_name': {'type': 'string', 'required': True},
                    'instance_type': {'type': 'string'},
                    'instance_count': {'type': 'integer', 'min': 1},
                },
            },
            'aws_lambda_operator_config': {
                'type': 'dict',
                'schema': {
                    'region': {'type': 'string', 'required': True},
                    'api_name': {'type': 'string'},
                },
            },
            'gcp_function_operator_config': {
                'type': 'dict',
                'schema': {
                    'region': {'type': 'string', 'required': True},
                    'api_name': {'type': 'string'},
                },
            },
            'kubernetes_operator_config': {
                'type': 'dict',
                'schema': {
                    'kube_namespace': {'type': 'string'},
                    'replicas': {'type': 'integer', 'min': 1},
                    'service_name': {'type': 'string'},
                    'service_type': {'type': 'string'},
                },
            },
        },
    },
    'state': {
        'type': 'dict',
        'schema': {
            'state': {'type': 'integer'},
            'error_message': {'type': 'string'},
            'info_json': {'type': 'string'},
        },
    },
}


def validate_pb_schema(pb, schema):
    pb_dict = ProtoMessageToDict(pb)
    v = Validator(schema)
    if v.validate(pb_dict):
        return None
    else:
        return v.errors


def validate_deployment_pb_schema(pb):
    return validate_pb_schema(pb, deployment_schema)
