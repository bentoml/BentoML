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
from bentoml.proto.deployment_pb2 import DeploymentSpec, DeploymentState

deployment_schema = {
    'name': {'type': 'string', 'required': True, 'minlength': 4},
    'namespace': {'type': 'string', 'required': True, 'minlength': 3},
    'labels': {'type': 'dict', 'allow_unknown': True},
    'annotations': {'type': 'dict', 'allow_unknown': True},
    'created_at': {'type': 'string'},
    'last_updated_at': {'type': 'string'},
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
            'bento_version': {
                'type': 'string',
                'required': True,
                'bento_service_version': True,
            },
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
                    'api_name': {'type': 'string', 'required': True, 'minlength': 3},
                    'instance_type': {'type': 'string', 'required': True},
                    'instance_count': {'type': 'integer', 'min': 1, 'required': True},
                    'region': {'type': 'string'},
                    'num_of_gunicorn_workers_per_instance': {
                        'type': 'integer',
                        'min': 1,
                    },
                },
            },
            'aws_lambda_operator_config': {
                'type': 'dict',
                'schema': {
                    'region': {'type': 'string'},
                    'api_name': {'type': 'string', 'minlength': 3},
                    'memory_size': {'type': 'integer', 'aws_lambda_memory': True},
                    'timeout': {'type': 'integer', 'min': 1, 'max': 900},
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
            'state': {'type': 'string', 'allowed': DeploymentState.State.keys()},
            'error_message': {'type': 'string'},
            'info_json': {'type': 'string'},
            'timestamp': {'type': 'string'},
        },
    },
}


class YataiDeploymentValidator(Validator):
    def _validate_aws_lambda_memory(self, aws_lambda_memory, field, value):
        """ Test the memory size restriction for AWS Lambda.

        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if aws_lambda_memory:
            if value > 3008 or value < 128:
                self._error(
                    field,
                    'AWS Lambda memory must be between 128 MB to 3,008 MB, '
                    'in 64 MB increments.',
                )
            if value % 64 > 0:
                self._error(
                    field,
                    'AWS Lambda memory must be between 128 MB to 3,008 MB, '
                    'in 64 MB increments.',
                )

    def _validate_bento_service_version(self, bento_service_version, field, value):
        """ Test the given BentoService version is not "latest"

        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if bento_service_version and value.lower() == "latest":
            self._error(
                field,
                'Must use specific "bento_version" in deployment, using "latest" is '
                'an anti-pattern.',
            )


def validate_pb_schema(pb, schema):
    pb_dict = ProtoMessageToDict(pb)
    v = YataiDeploymentValidator(schema)
    if v.validate(pb_dict):
        return None
    else:
        return v.errors


def validate_deployment_pb_schema(pb):
    return validate_pb_schema(pb, deployment_schema)
