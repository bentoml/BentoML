# pylint: disable=redefined-outer-name

import contextlib
import os

import pytest

from tests.integration.utils import (
    build_api_server_docker_image,
    run_api_server,
    run_api_server_docker_container,
)


def pytest_addoption(parser):
    parser.addoption("--bento", action="store", default="service:svc")
    parser.addoption("--project-root", action="store", default="./")
    parser.addoption("--docker", action="store_true", help="run in docker")
    parser.addoption(
        "--config-file", action="store", help="local configuration file", default=None
    )
    parser.addoption(
        "--dev-server",
        action="store_true",
        help="run with development server, available on Windows",
    )


@pytest.fixture(scope="session")
def clean_context():
    with contextlib.ExitStack() as stack:
        yield stack


@pytest.fixture(scope="session")
def host(pytestconfig, clean_context):
    """
    Launch host from a
    """
    test_workdir = pytestconfig.getoption("project_root")
    test_svc_bundle = pytestconfig.getoption("bento")

    config_file = pytestconfig.getoption("config_file") or os.path.join(
        test_workdir, "bentoml_config.yml"
    )
    if not os.path.exists(config_file):
        if pytestconfig.getoption("config_file"):
            raise Exception(f"config file not found: {config_file}")
        else:
            config_file = None

    if pytestconfig.getoption("docker"):
        image = clean_context.enter_context(
            build_api_server_docker_image(test_svc_bundle, "example_service")
        )
        with run_api_server_docker_container(
            image,
            config_file=config_file,
        ) as host:
            yield host
    elif pytestconfig.getoption("dev_server"):
        with run_api_server(
            test_svc_bundle,
            config_file=config_file,
            workdir=test_workdir,
            dev_server=True,
        ) as host:
            yield host
    else:
        with run_api_server(
            test_svc_bundle,
            config_file=config_file,
            workdir=test_workdir,
        ) as host:
            yield host


@pytest.fixture(scope="session")
def service(pytestconfig):
    test_svc_bundle = pytestconfig.getoption("bento")

    import bentoml

    return bentoml.load(test_svc_bundle)
