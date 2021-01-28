from sys import version_info

import pytest

from mock import patch

from bentoml.yatai.deployment.docker_utils import ensure_docker_available_or_raise
from bentoml.exceptions import MissingDependencyException


def raise_(ex):
    raise ex


def test_ensure_docker_available_or_raise():
    if version_info.major < 3:
        not_found_error = OSError
    else:
        not_found_error = FileNotFoundError

    with patch('subprocess.check_output', new=lambda x: raise_(not_found_error())):
        with pytest.raises(MissingDependencyException) as error:
            ensure_docker_available_or_raise()
        assert str(error.value).startswith('Docker is required')
