import pytest

from bentoml.exceptions import ImportServiceError
from bentoml._internal.service.loader import import_service


@pytest.mark.usefixtures("change_test_dir")
def test_load_by_service_file():
    svc = import_service("bento_service_in_module/single_service.py")
    assert svc.name == "test-bento-service-in-module-single"

    svc = import_service("./bento_service_in_module/single_service.py:svc")
    assert svc.name == "test-bento-service-in-module-single"

    svc = import_service("./single_service_in_package/__init__.py")
    assert svc.name == "test-bento-service-in-package"

    svc = import_service("./single_service_in_package/__init__.py:svc")
    assert svc.name == "test-bento-service-in-package"


@pytest.mark.usefixtures("change_test_dir")
def test_load_by_module_name():
    svc = import_service("bento_service_in_module.single_service")
    assert svc.name == "test-bento-service-in-module-single"

    svc = import_service("bento_service_in_module.single_service:svc")
    assert svc.name == "test-bento-service-in-module-single"

    svc = import_service("single_service_in_package")
    assert svc.name == "test-bento-service-in-package"

    svc = import_service("single_service_in_package:svc")
    assert svc.name == "test-bento-service-in-package"


@pytest.mark.usefixtures("change_test_dir")
def test_load_multi_service_module():
    svc = import_service("bento_service_in_module.multi_service:svc_i")
    assert svc.name == "test-bento-service-in-module-i"

    svc = import_service("bento_service_in_module.multi_service:svc_ii")
    assert svc.name == "test-bento-service-in-module-ii"

    with pytest.raises(
        ImportServiceError, match="Multiple Service instances found in module"
    ):
        import_service("bento_service_in_module.multi_service")

    svc = import_service("multi_service_in_package:svc_i")
    assert svc.name == "test-multi-service-in-package-i"

    svc = import_service("multi_service_in_package:svc_ii")
    assert svc.name == "test-multi-service-in-package-ii"

    with pytest.raises(
        ImportServiceError, match="Multiple Service instances found in module"
    ):
        import_service("multi_service_in_package")
