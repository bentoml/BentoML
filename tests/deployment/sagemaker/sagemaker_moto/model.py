import time

from moto.core import BaseBackend
from moto.ec2 import ec2_backends
from moto.iam.models import ACCOUNT_ID


BASE_SAGEMAKER_ARN = 'arn:aws:sagemaker:{region}:{account}'

ENDPOINT_STATUS = {
    'InService': 'InService',
    'Creating': 'Creating',
    'Updating': 'Updating',
    'Failed': 'Failed',
}


class EndpointOperation:
    def __init__(self, created_time, latency_seconds, pending_state, complete_state):
        self.start_time = time.time()
        self.created_time = created_time
        self.pending_state = (pending_state,)
        self.complete_state = (complete_state,)
        self.latency_seconds = latency_seconds

    def status(self):
        if time.time() - self.created_time < self.latency_seconds:
            return self.pending_state
        else:
            return self.complete_state

    @classmethod
    def create_successful(cls, created_time, latency_seconds):
        return cls(
            created_time,
            latency_seconds,
            ENDPOINT_STATUS['Creating'],
            ENDPOINT_STATUS['InService'],
        )

    @classmethod
    def create_unsuccessful(cls, created_time, latency_seconds):
        return cls(
            created_time,
            latency_seconds,
            ENDPOINT_STATUS['Creating'],
            ENDPOINT_STATUS['Failed'],
        )

    @classmethod
    def update_successful(cls, created_time, latency_seconds):
        return cls(
            created_time,
            latency_seconds,
            ENDPOINT_STATUS['Updating'],
            ENDPOINT_STATUS['InService'],
        )

    @classmethod
    def update_unsuccessful(cls, created_time, latency_seconds):
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

    def generate_arn(self, region_name, info=''):
        base_arn = BASE_SAGEMAKER_ARN.format(region=region_name, account=ACCOUNT_ID)
        return base_arn + info

    def describe_endpoint(self, endpoint_name):
        if endpoint_name not in self.endpoints:
            raise ValueError()

        endpoint = self.endpoints[endpoint_name]
        config = self.endpoint_configs[endpoint.resource.config_name]
        return {
            'Endpoint': endpoint['EndpointName'],
            'EndpointConfigName': endpoint['EndpointConfigName'],
            'EndpointArn': endpoint['arn'],
            'EndpointStatus': endpoint['status'],
            'CreationTime': endpoint['creation_time'],
            'LastModifiedTime': endpoint['last_modified_time'],
            'ProductionVariants': config.production_variants,
        }

    def delete_endpoint(self, endpoint_name):
        if endpoint_name not in self.endpoints:
            raise ValueError('')

        del self.endpoints[endpoint_name]

    def delete_endpoint_config(self, config_name):
        if config_name not in self.endpoint_configs:
            raise ValueError('')
        del self.endpoint_configs[config_name]

    def delete_model(self, model_name):
        if model_name not in self.models:
            raise ValueError('')

        del self.models[model_name]

    def create_model(
        self, model_name, tags, primary_container, execution_role_arn, region_name
    ):
        if model_name in self.models:
            raise ValueError('')
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
        info = {}
        return {
            'resource': info,
            'arn': self.generate_arn(
                region_name, ':endpoint-config/{}'.format(endpoint_config_name)
            ),
        }

    def create_endpoint(self, endpoint_name, endpoint_config_name, region_name):
        if endpoint_name in self.endpoints:
            raise ValueError('')

        if endpoint_config_name not in self.endpoint_configs:
            raise ValueError('')

        info = {
            'EndpointName': endpoint_name,
            'EndpointConfigName': endpoint_config_name,
            'arn': self.generate_arn(region_name, ':endpoint/{}'.format(endpoint_name)),
            'created_time': time.time(),
        }
        info = EndpointOperation.create_successful(
            info['created_time'], latency_seconds=60
        )

        self.endpoints[endpoint_name] = info
        return {'resource': info, 'arn': info['arn']}

    def update_endpoint(self, endpoint_name, config_name):
        if endpoint_name not in self.endpoints:
            raise ValueError('')
        if config_name not in self.endpoint_configs:
            raise ValueError('')
        endpoint = self.endpoints[endpoint_name]
        endpoint.resource.config_name = config_name

        return endpoint


sagemaker_backends = {}

for region, _ in ec2_backends.items():
    sagemaker_backends[region] = SageMakerBackend()
