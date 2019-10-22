import pytest

from subprocess import CalledProcessError
import subprocess

from bentoml.deployment.utils import ensure_docker_available_or_raise
from bentoml.exceptions import BentoMLMissingDependencyException, BentoMLException


def raise_(ex):
    raise ex


def test_ensure_docker_available_or_raise():
    setattr(
        subprocess, 'check_output', lambda *args, **kwargs: raise_(FileNotFoundError())
    )
    with pytest.raises(BentoMLMissingDependencyException) as error:
        ensure_docker_available_or_raise()
    assert str(error.value).startswith('Docker is required')

    setattr(
        subprocess,
        'check_output',
        lambda *args, **kwargs: raise_(
            CalledProcessError('fake_return_code', 'fake_cmd')
        ),
    )
    with pytest.raises(BentoMLException) as error:
        ensure_docker_available_or_raise()
    assert str(error.value).startswith('Error executing docker command:')
