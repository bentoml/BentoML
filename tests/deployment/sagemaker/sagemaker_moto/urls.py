from tests.deployment.sagemaker.sagemaker_moto.response import SageMakerResponse


url_bases = ['https?://api.sagemaker.(.+).amazonaws.com']

url_paths = {'{0}/$': SageMakerResponse.dispatch}
