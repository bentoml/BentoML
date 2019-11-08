import json

from moto.core.responses import BaseResponse

from tests.deployment.sagemaker.sagemaker_moto.model import sagemaker_backends


class SageMakerResponse(BaseResponse):
    """SageMaker response for moto mock.
    References API operations and result from
    https://docs.aws.amazon.com/sagemaker/latest/dg/API_Operations.html
    """

    @property
    def request_body(self):
        return json.loads(self.body)

    @property
    def backend(self):
        return sagemaker_backends[self.region]

    def create_model(self):
        model_name = self.request_body['ModelName']
        tags = self.request_body.get('Tags', [])
        primary_container = self.request_body['PrimaryContainer']
        execution_role_arn = self.request_body['ExecutionRoleArn']

        result = self.backend.create_model(
            model_name, tags, primary_container, execution_role_arn, self.region
        )
        return json.dumps({'ModelArn': result['arn']})

    def create_endpoint_config(self):
        config_name = self.request_body['EndpointConfigName']
        production_variants = self.request_body['ProductionVariants']
        result = self.backend.create_endpoint_config(
            config_name, production_variants, self.region
        )
        return json.dumps({'EndpointConfigArn': result['arn']})

    def create_endpoint(self):
        endpoint_name = self.request_body['EndpointName']
        config_name = self.request_body['EndpointConfigName']
        result = self.backend.create_endpoint(endpoint_name, config_name, self.region)
        return json.dumps({'EndpointArn': result['arn']})

    def describe_endpoint(self):
        endpoint_name = self.request_body['EndpointName']
        endpoint_description = self.backend.describe_endpoint(endpoint_name)
        return json.dumps(endpoint_description)

    def delete_model(self):
        model_name = self.request_body['ModelName']
        self.backend.delete_model(model_name)
        return ''

    def delete_endpoint_config(self):
        config_name = self.request_body['EndpointConfigName']
        self.backend.delete_endpoint_config(config_name)
        return ''

    def delete_endpoint(self):
        endpoint_name = self.request_body['EndpointName']
        self.backend.delete_endpoint(endpoint_name)
        return ''

    def update_endpoint(self):
        endpoint_name = self.request_body['EndpointName']
        config_name = self.request_body['EndpointConfigName']
        result = self.backend.update_endpoint(endpoint_name, config_name)
        return json.dumps({'EndpointArn': result['arn']})
