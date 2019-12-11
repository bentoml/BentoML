from mock import patch
import pytest

from bentoml.deployment.serverless.serverless_utils import (
    check_nodejs_compatible_version,
)
from bentoml.exceptions import MissingDependencyException


def test_check_nodejs_compatible_version():
    with pytest.raises(MissingDependencyException) as error:
        with patch('bentoml.utils.whichcraft.which', return_value=None):
            check_nodejs_compatible_version()
    assert str(error.value).startswith('NPM is not installed')
