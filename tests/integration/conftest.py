# pylint: disable=redefined-outer-name
import contextlib

import pytest


@pytest.fixture(scope="session")
def clean_context():
    with contextlib.ExitStack() as stack:
        yield stack


@pytest.fixture(params=[True, False], scope="module")
def enable_microbatch(request):
    pytest.enable_microbatch = request.param
    return request.param
