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

from cerberus import Validator

from bentoml.exceptions import InvalidArgument
from bentoml.utils import ProtoMessageToDict
from bentoml.yatai.db.stores.label import _validate_labels
from bentoml.yatai.deployment.azure_functions.constants import (
    AZURE_FUNCTIONS_PREMIUM_PLAN_SKUS,
    AZURE_FUNCTIONS_AUTH_LEVELS,
)
from bentoml.yatai.proto.deployment_pb2 import DeploymentSpec, DeploymentState

deployment_schema = {
    'name': {'type': 'string', 'required': True, 'minlength': 4},
    # namespace is optional - YataiService will fill-in the default namespace configured
    # when it is missing in the apply deployment request
    'namespace': {'type': 'string', 'required': False, 'minlength': 3},
    'labels': {'type': 'dict', 'deployment_labels': True},
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
                    'timeout': {'type': 'integer', 'min': 1},
                    'data_capture_s3_prefix': {'type': 'string'},
                    'data_capture_sample_percent': {
                        'type': 'integer',
                        'min': 1,
                        'max': 100,
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
            # https://docs.microsoft.com/en-us/azure/azure-functions/functions-premium-plan
            # https://docs.microsoft.com/en-us/azure/azure-functions/functions-bindings-http-webhook-trigger?tabs=python#configuration
            'azure_functions_operator_config': {
                'type': 'dict',
                'azure_functions_configuration': True,
                'schema': {
                    'location': {'type': 'string'},
                    'premium_plan_sku': {
                        'type': 'string',
                        'allowed': AZURE_FUNCTIONS_PREMIUM_PLAN_SKUS,
                    },
                    'min_instances': {
                        'required': True,
                        'type': 'integer',
                        'min': 1,
                        'max': 20,
                    },
                    'max_burst': {
                        'type': 'integer',
                        'min': 1,
                        'max': 20,
                        'required': True,
                    },
                    'function_auth_level': {
                        'type': 'string',
                        'allowed': AZURE_FUNCTIONS_AUTH_LEVELS,
                    },
                },
            },
            "aws_ec2_operator_config": {
                "type": "dict",
                "aws_ec2_operator_configurations": True,
                "schema": {
                    "region": {"type": "string"},
                    "instance_type": {"type": "string"},
                    "ami_id": {"type": "string"},
                    "autoscale_min_size": {
                        "type": "integer",
                        "min": 0,
                        "required": True,
                    },
                    "autoscale_desired_capacity": {
                        "type": "integer",
                        "min": 0,
                        "required": True,
                    },
                    "autoscale_max_size": {
                        "type": "integer",
                        "min": 0,
                        "required": True,
                    },
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

    def _validate_azure_functions_configuration(
        self, azure_functions_configuration, field, value
    ):
        """ Test the min instances is less than max burst for Azure Functions deployment

        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if azure_functions_configuration:
            if value.get('max_burst', 0) < value.get('min_instances', 0):
                self._error(
                    field,
                    'Azure Functions min instances must be smaller than max burst',
                )

    def _validate_deployment_labels(self, deployment_labels, field, value):
        """ Test label key value schema

        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if deployment_labels:
            try:
                _validate_labels(value)
            except InvalidArgument:
                self._error(
                    field,
                    'Valid label key and value must be 63 characters or less and '
                    'must be being and end with an alphanumeric character '
                    '[a-z0-9A-Z] with dashes (-), underscores (_), and dots (.)',
                )

    def _validate_aws_ec2_operator_configurations(
        self, aws_ec2_operator_configurations, field, value
    ):
        """ Test label key value schema

        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        if aws_ec2_operator_configurations:
            if (
                value.get("autoscale_min_size") < 0
                or value.get("autoscale_max_size") < value.get("autoscale_min_size")
                or value.get("autoscale_desired_capacity")
                < value.get("autoscale_min_size")
                or value.get("autoscale_desired_capacity")
                > value.get("autoscale_max_size")
            ):
                self._error(
                    field,
                    "Wrong autoscaling size specified. "
                    "It should be min_size <= desired_capacity <= max_size",
                )


def validate_deployment_pb(pb):
    pb_dict = ProtoMessageToDict(pb)
    v = YataiDeploymentValidator(deployment_schema)
    if v.validate(pb_dict):
        return None
    else:
        return v.errors
