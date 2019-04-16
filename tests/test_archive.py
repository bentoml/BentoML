import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bentoml.archive.archiver import _validate_version_str  # noqa: E402

def test_validate_version_str_fails():
    with pytest.raises(ValueError):
        _validate_version_str('44&')

def test_validate_version_str_pass():
    _validate_version_str('abc_123')
