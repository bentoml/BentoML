import time

from moto.core import BaseBackend
from moto.ec2 import ec2_backends
from moto.iam.models import ACCOUNT_ID
from botocore.exceptions import ClientError

BASE_SAGEMAKER_ARN = 'arn:aws:sagemaker:{region}:{account}'

ENDPOINT_STATUS = {
    'InService': 'InService',
    'Creating': 'Creating',
    'Updating': 'Updating',
    'Failed': 'Failed',
}

DEFAULT_ENDPOINT_OPERATION_LATENCY_SECONDS = 60


class EndpointOperation:
    def __init__(self, created_time, latency_seconds, pending_state, complete_state):
        self.start_time = time.time()
        self.created_time = created_time
        self.pending_state = pending_state
        self.complete_state = complete_state
        self.latency_seconds = latency_seconds

    def status(self):
        if time.time() - self.created_time < self.latency_seconds:
            return self.pending_state
        else:
            return self.complete_state

    @classmethod
    def create_successful(
        cls, created_time, latency_seconds=DEFAULT_ENDPOINT_OPERATION_LATENCY_SECONDS
    ):
        return cls(
            created_time,
            latency_seconds,
            ENDPOINT_STATUS['Creating'],
            ENDPOINT_STATUS['InService'],
        )

    @classmethod
    def create_unsuccessful(
        cls, created_time, latency_seconds=DEFAULT_ENDPOINT_OPERATION_LATENCY_SECONDS
    ):
        return cls(
            created_time,
            latency_seconds,
            ENDPOINT_STATUS['Creating'],
            ENDPOINT_STATUS['Failed'],
        )

    @classmethod
    def update_successful(
        cls, created_time, latency_seconds=DEFAULT_ENDPOINT_OPERATION_LATENCY_SECONDS
    ):
        return cls(
            created_time,
            latency_seconds,
            ENDPOINT_STATUS['Updating'],
            ENDPOINT_STATUS['InService'],
        )

    @classmethod
    def update_unsuccessful(
        cls, created_time, latency_seconds=DEFAULT_ENDPOINT_OPERATION_LATENCY_SECONDS
    ):
        return cls(
            created_time,
            latency_seconds,
            ENDPOINT_STATUS['Updating'],
            ENDPOINT_STATUS['Failed'],
        )


class SageMakerBackend(BaseBackend):
    """SageMaker service for moto mock.
    References API operations and result from
    https://docs.aws.amazon.com/sagemaker/latest/dg/API_Operations.html
    """

    def __init__(self):
        self.models = {}
        self.endpoints = {}
        self.endpoint_configs = {}

    @property
    def _url_module(self):
        urls_module = __import__(
            'tests.deployment.sagemaker.sagemaker_moto.urls',
            fromlist=['url_bases', 'url_paths'],
        )
        return urls_module

    @staticmethod
    def generate_arn(region_name, info=''):
        base_arn = BASE_SAGEMAKER_ARN.format(region=region_name, account=ACCOUNT_ID)
        return base_arn + info

    def describe_endpoint(self, endpoint_name):
        if endpoint_name not in self.endpoints:
            raise ValueError('Endpoint {} does not exist'.format(endpoint_name))

        endpoint = self.endpoints[endpoint_name]
        config = self.endpoint_configs[endpoint['EndpointConfigName']]
        return {
            'Endpoint': endpoint['EndpointName'],
            'EndpointConfigName': endpoint['EndpointConfigName'],
            'EndpointArn': endpoint['arn'],
            'EndpointStatus': endpoint['latest_operation'].status(),
            'CreationTime': endpoint['created_time'],
            'LastModifiedTime': endpoint['latest_operation'].created_time,
            'ProductionVariants': config['ProductionVariants'],
        }

    def delete_endpoint(self, endpoint_name):
        if endpoint_name not in self.endpoints:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "Could not find endpoint ...",
                    }
                },
                "DeleteEndpoint",
            )

        del self.endpoints[endpoint_name]

    def delete_endpoint_config(self, config_name):
        if config_name not in self.endpoint_configs:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "Could not find endpoint configuration ...",
                    }
                },
                "DeleteEndpointConfiguration",
            )
        del self.endpoint_configs[config_name]

    def delete_model(self, model_name):
        if model_name not in self.models:
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "Could not find model ...",
                    }
                },
                "DeleteModel",
            )
        del self.models[model_name]

    def create_model(
        self, model_name, tags, primary_container, execution_role_arn, region_name
    ):
        if model_name in self.models:
            raise ValueError('Model {} already exists'.format(model_name))
        info = {
            "Containers": [primary_container],
            'PrimaryContainer': primary_container,
            'ExecutionRoleArn': execution_role_arn,
            'Tags': tags,
        }
        self.models[model_name] = info
        return {
            'resource': info,
            'arn': self.generate_arn(region_name, ':model/{}'.format(model_name)),
        }

    def create_endpoint_config(
        self, endpoint_config_name, production_variants, region_name
    ):
        if endpoint_config_name in self.endpoint_configs:
            raise ValueError(
                'Endpoint configuration {} already exists'.format(endpoint_config_name)
            )
        for production_variant in production_variants:
            if "ModelName" not in production_variant:
                raise ValueError('ModelName is required for ProductionVariants')
            elif production_variant['ModelName'] not in self.models:
                raise ValueError(
                    'Model {} does not exist'.format(production_variant['ModelName'])
                )
            model = self.models[production_variant['ModelName']]
            production_variant['DeployedImages'] = [
                {'SpecifiedImage': model['PrimaryContainer']['Image']}
            ]

        info = {
            'EndpointConfigName': endpoint_config_name,
            'ProductionVariants': production_variants,
        }
        self.endpoint_configs[endpoint_config_name] = info
        return {
            'resource': info,
            'arn': self.generate_arn(
                region_name, ':endpoint-config/{}'.format(endpoint_config_name)
            ),
        }

    def create_endpoint(self, endpoint_name, endpoint_config_name, region_name):
        if endpoint_name in self.endpoints:
            raise ValueError('Endpoint {} already exists'.format(endpoint_name))

        if endpoint_config_name not in self.endpoint_configs:
            raise ValueError(
                'Endpoint configuration {} does not exist'.format(endpoint_config_name)
            )

        created_time = time.time()
        info = {
            'EndpointName': endpoint_name,
            'EndpointConfigName': endpoint_config_name,
            'arn': self.generate_arn(region_name, ':endpoint/{}'.format(endpoint_name)),
            'created_time': created_time,
            'latest_operation': EndpointOperation.create_successful(created_time),
        }

        self.endpoints[endpoint_name] = info
        return {'resource': info, 'arn': info['arn']}

    def update_endpoint(self, endpoint_name, config_name):
        if endpoint_name not in self.endpoints:
            raise ValueError('Endpoint {} does not exist'.format(endpoint_name))
        if config_name not in self.endpoint_configs:
            raise ValueError(
                'Endpoint configuration {} does not exist'.format(config_name)
            )
        endpoint = self.endpoints[endpoint_name]
        endpoint['EndpointConfigName'] = config_name
        endpoint['latest_operation'] = EndpointOperation.update_successful(time.time())

        return {'arn': endpoint['arn']}


sagemaker_backends = {}

for region, _ in ec2_backends.items():
    sagemaker_backends[region] = SageMakerBackend()
