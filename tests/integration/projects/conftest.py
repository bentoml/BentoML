# pylint: disable=redefined-outer-name

import contextlib
import os
import sys

import pytest

from tests import (
    build_api_server_docker_image,
    run_api_server,
    run_api_server_docker_container,
)


def pytest_addoption(parser):
    parser.addoption("--bento-dist", action="store", default=None)
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
    test_svc_bundle = pytestconfig.getoption("bento_dist") or os.path.join(
        sys.argv[2], "build", "dist"
    )

    config_file = pytestconfig.getoption("config_file") or os.path.join(
        sys.argv[2], "bentoml_config.yml"
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
            dev_server=True,
        ) as host:
            yield host
    else:
        with run_api_server(
            test_svc_bundle,
            config_file=config_file,
        ) as host:
            yield host


@pytest.fixture(scope="session")
def service(pytestconfig):
    test_svc_bundle = pytestconfig.getoption("bento_dist") or os.path.join(
        sys.argv[1], "build", "dist"
    )

    import bentoml

    return bentoml.load_from_dir(test_svc_bundle)
