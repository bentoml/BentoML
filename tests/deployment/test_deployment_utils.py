from sys import version_info

import pytest

from subprocess import CalledProcessError
import subprocess

from bentoml.deployment.utils import ensure_docker_available_or_raise
from bentoml.exceptions import BentoMLMissingDependencyException, BentoMLException


def raise_(ex):
    raise ex


def test_ensure_docker_available_or_raise():
    if version_info.major < 3:
        not_found_error = OSError
    else:
        not_found_error = FileNotFoundError

    setattr(
        subprocess, 'check_output', lambda *args, **kwargs: raise_(not_found_error())
    )
    with pytest.raises(BentoMLMissingDependencyException) as error:
        ensure_docker_available_or_raise()
    assert str(error.value).startswith('Docker is required')

    setattr(
        subprocess,
        'check_output',
        lambda *args, **kwargs: raise_(CalledProcessError(2, 'fake_cmd')),
    )
    with pytest.raises(BentoMLException) as error:
        ensure_docker_available_or_raise()
    assert str(error.value).startswith('Error executing docker command:')
