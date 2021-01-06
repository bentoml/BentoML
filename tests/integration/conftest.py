# pylint: disable=redefined-outer-name
import contextlib

import pytest

from tests.integration.utils import (
    build_api_server_docker_image,
    export_service_bundle,
    run_api_server_docker_container,
)


@pytest.fixture(scope="session")
def clean_context():
    with contextlib.ExitStack() as stack:
        yield stack


@pytest.fixture(params=[True, False], scope="session")
def batch_mode(request):
    return request.param


@pytest.fixture(scope="session")
def image(svc, clean_context):
    with export_service_bundle(svc) as saved_path:
        yield clean_context.enter_context(build_api_server_docker_image(saved_path))


@pytest.fixture(params=[True, False], scope="module")
def enable_microbatch(request):
    pytest.enable_microbatch = request.param
    return request.param


@pytest.fixture(scope="module")
def host(image, enable_microbatch):
    with run_api_server_docker_container(
        image, enable_microbatch=enable_microbatch, timeout=500
    ) as host:
        yield host
