# pylint: disable=redefined-outer-name
import logging

import pytest

from tests.integration.utils import (
    build_api_server_docker_image,
    export_service_bundle,
    run_api_server,
    run_api_server_docker_container,
)

from .example_service import ExampleBentoService, ExampleBentoServiceSingle, PickleModel

logger = logging.getLogger("bentoml.tests")


def pytest_addoption(parser):
    parser.addoption("--docker", action="store_true")
    parser.addoption("--bentomlver", action="store", default="")


@pytest.fixture(scope="session")
def with_docker(pytestconfig):
    return pytestconfig.getoption("docker")


@pytest.fixture(scope="session")
def bentoml_version(pytestconfig):
    return pytestconfig.getoption("bentomlver")


@pytest.fixture(params=[True, False], scope="session")
def batch_mode(request):
    return request.param


@pytest.fixture(scope="session")
def test_svc(batch_mode):

    # When the ExampleBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    if batch_mode:
        svc_cls = ExampleBentoService
    else:
        svc_cls = ExampleBentoServiceSingle
    svc_cls._bento_service_bundle_path = None
    svc_cls._bento_service_bundle_version = None
    test_svc = svc_cls()

    pickle_model = PickleModel()
    test_svc.pack('model', pickle_model)

    from sklearn.ensemble import RandomForestRegressor

    sklearn_model = RandomForestRegressor(n_estimators=2)
    sklearn_model.fit(
        [[i] for _ in range(100) for i in range(10)],
        [i for _ in range(100) for i in range(10)],
    )
    test_svc.pack('sk_model', sklearn_model)
    return test_svc


@pytest.fixture(scope="session")
def test_svc_bundle(clean_context, test_svc):
    return clean_context.enter_context(export_service_bundle(test_svc))


@pytest.fixture(scope="module")
def host(
    clean_context, test_svc_bundle, enable_microbatch, with_docker, bentoml_version
):
    if with_docker:
        image = clean_context.enter_context(
            build_api_server_docker_image(test_svc_bundle, "example_service")
        )
        with run_api_server_docker_container(image, enable_microbatch) as host:
            yield host
    else:
        with run_api_server(
            test_svc_bundle, enable_microbatch, use_bentoml_package=bentoml_version
        ) as host:
            yield host
