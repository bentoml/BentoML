# pylint: disable=redefined-outer-name

import contextlib
import os
import sys

import pytest

from tests.integration.utils import (
    build_api_server_docker_image,
    run_api_server,
    run_api_server_docker_container,
)


def pytest_addoption(parser):
    parser.addoption("--bento-dist", action="store", default=None)
    parser.addoption("--docker", action="store_true", help="run in docker")


@pytest.fixture(scope="session")
def clean_context():
    with contextlib.ExitStack() as stack:
        yield stack


@pytest.fixture(scope="session")
def bundle(pytestconfig):
    test_svc_bundle = pytestconfig.getoption("bento_dist") or os.path.join(
        sys.argv[1], "build", "dist"
    )
    return test_svc_bundle


@pytest.fixture(params=[True, False], scope="module")
def enable_microbatch(request):
    pytest.enable_microbatch = request.param
    return request.param


@pytest.fixture(scope="module")
def host(pytestconfig, bundle, clean_context, enable_microbatch):
    if pytestconfig.getoption("docker"):
        image = clean_context.enter_context(
            build_api_server_docker_image(bundle, "example_service")
        )
        with run_api_server_docker_container(image, enable_microbatch) as host:
            yield host
    else:
        with run_api_server(bundle, enable_microbatch) as host:
            yield host


@pytest.fixture(scope="module")
def service(bundle):
    import bentoml

    return bentoml.load_from_dir(bundle)
