from moto.core.models import base_decorator

from tests.deployment.sagemaker.sagemaker_moto.model import sagemaker_backends

moto_mock_sagemaker = base_decorator(sagemaker_backends)
