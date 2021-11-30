import os

import pytest


@pytest.fixture(scope="function", name="change_test_dir")
def fixture_change_test_dir(request):
    os.chdir(request.fspath.dirname)
    yield
    os.chdir(request.config.invocation_dir)
