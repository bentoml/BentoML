from moto.core.models import base_decorator

from tests.deployment.sagemaker.sagemaker_moto.model import sagemaker_backends
from tests.deployment.sagemaker.sagemaker_moto.responses import SageMakerResponse


moto_mock_sagemaker = base_decorator(sagemaker_backends)

url_bases = ['https?://api.sagemaker.(.+).amazonaws.com']

url_paths = {'{0}/$': SageMakerResponse.dispatch}
