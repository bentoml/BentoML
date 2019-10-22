import pytest

from bentoml.deployment.utils import ensure_docker_available_or_raise
from bentoml.exceptions import BentoMLMissingDependencyException, BentoMLException


def test_ensure_docker_available_or_raise():
    with pytest.raises(BentoMLException) as error:
        ensure_docker_available_or_raise()
    if type(error) == BentoMLMissingDependencyException:
        assert str(error.value).startswith('Docker is required for this')
    else:
        assert str(error.value).startswith('Error executing docker command:')
