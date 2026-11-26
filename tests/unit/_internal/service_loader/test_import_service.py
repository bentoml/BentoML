import os
import sys

import pytest

from bentoml._internal.service.loader import import_service
from bentoml._internal.service.loader import load
from bentoml.exceptions import ImportServiceError


@pytest.fixture
def conflicting_service_projects(tmp_path, monkeypatch):
    module = "working_dir_priority_service"
    projects = []
    for name in ("first_project", "target_project"):
        project = tmp_path / name
        project.mkdir()
        (project / f"{module}.py").write_text(
            f"from bentoml.legacy import Service\nsvc = Service({name!r})\n"
        )
        projects.append(project)
    monkeypatch.setattr(sys, "path", [str(projects[0]), *sys.path, str(projects[1])])
    yield module, projects
    sys.modules.pop(module, None)


@pytest.mark.parametrize("loader", [import_service, load])
@pytest.mark.parametrize("path_position", ["absent", "later", "first"])
def test_working_dir_service_priority(
    conflicting_service_projects, loader, path_position
):
    module, (first, target) = conflicting_service_projects
    target_path = str(target)
    if path_position == "absent":
        sys.path.remove(target_path)
    elif path_position == "first":
        sys.path.remove(target_path)
        sys.path.insert(0, target_path)
    original_path, original_cwd = list(sys.path), os.getcwd()

    svc = loader(f"{module}:svc", working_dir=target_path)

    assert svc.name == "target_project"
    assert sys.modules[module].__file__ == str(target / f"{module}.py")
    assert sys.path == original_path
    assert os.getcwd() == original_cwd


def test_working_dir_priority_preserves_cached_service(conflicting_service_projects):
    module, (first, target) = conflicting_service_projects
    cached = import_service(f"{module}:svc", working_dir=str(first))
    original_path, original_cwd = list(sys.path), os.getcwd()

    assert import_service(f"{module}:svc", working_dir=str(target)) is cached
    assert sys.path == original_path
    assert os.getcwd() == original_cwd

    reloaded = import_service(f"{module}:svc", working_dir=str(target), reload=True)
    assert reloaded.name == "target_project"
    assert sys.path == original_path
    assert os.getcwd() == original_cwd


def test_bento_build_uses_working_dir_service(conflicting_service_projects):
    from bentoml._internal.bento import Bento
    from bentoml._internal.bento.build_config import BentoBuildConfig

    module, (_, target) = conflicting_service_projects
    original_path, original_cwd = list(sys.path), os.getcwd()
    config = BentoBuildConfig(service=f"{module}:svc")

    bento = Bento.create(
        config, version="priority-test", build_ctx=str(target), bare=True
    )

    assert bento.tag.name == "target_project"
    assert sys.path == original_path
    assert os.getcwd() == original_cwd


@pytest.mark.parametrize("missing", ["module", "attribute"])
def test_working_dir_priority_error_restores_context(
    conflicting_service_projects, tmp_path, missing
):
    from bentoml._internal.configuration.containers import BentoMLContainer
    from bentoml._internal.models import ModelStore

    module, (_, target) = conflicting_service_projects
    original_path, original_cwd = list(sys.path), os.getcwd()
    original_store = BentoMLContainer.model_store.get()
    model_store = ModelStore(str(tmp_path / "model_store"))
    import_target = (
        "absent_priority_test_module:svc" if missing == "module" else f"{module}:absent"
    )

    with pytest.raises(ImportServiceError):
        import_service(import_target, working_dir=str(target), model_store=model_store)

    assert sys.path == original_path
    assert os.getcwd() == original_cwd
    assert BentoMLContainer.model_store.get() is original_store


@pytest.mark.parametrize(
    ("source", "error"),
    [
        ('raise ValueError("service failed")\n', ValueError),
        ("def invalid syntax\n", SyntaxError),
        ('raise SystemExit("import aborted")\n', SystemExit),
    ],
)
@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("already_listed", [False, True])
def test_failed_import_restores_context(
    conflicting_service_projects, tmp_path, source, error, standalone, already_listed
):
    from bentoml._internal.configuration.containers import BentoMLContainer
    from bentoml._internal.models import ModelStore

    module, (_, target) = conflicting_service_projects
    (target / f"{module}.py").write_text(source)
    if not already_listed:
        sys.path.remove(str(target))
    original_path, original_cwd = list(sys.path), os.getcwd()
    original_store = BentoMLContainer.model_store.get()
    model_store = ModelStore(str(tmp_path / "failed_import_store"))
    try:
        with pytest.raises(error):
            import_service(
                f"{module}:svc",
                working_dir=str(target),
                standalone_load=standalone,
                model_store=model_store,
            )
        assert sys.path == original_path
        assert os.getcwd() == original_cwd
        assert BentoMLContainer.model_store.get() is original_store
    finally:
        # Keep a failing regression from affecting the rest of the loader tests.
        sys.path[:] = original_path
        os.chdir(original_cwd)
        BentoMLContainer.model_store.set(original_store)


def test_standalone_working_dir_priority(conflicting_service_projects, monkeypatch):
    module, (_, target) = conflicting_service_projects
    original_cwd = os.getcwd()
    # Restore the process after checking the intentionally persistent standalone load.
    monkeypatch.chdir(original_cwd)

    svc = import_service(f"{module}:svc", working_dir=str(target), standalone_load=True)

    assert svc.name == "target_project"
    assert os.getcwd() == str(target)
    assert sys.path[0] == str(target)
    assert sys.path.count(str(target)) == 1
    path_after_load = list(sys.path)
    assert (
        import_service(f"{module}:svc", working_dir=str(target), standalone_load=True)
        is svc
    )
    assert sys.path == path_after_load


def test_working_dir_priority_preserves_cached_dependency(
    conflicting_service_projects,
):
    import schema

    module, (_, target) = conflicting_service_projects
    (target / "schema.py").write_text(
        'raise AssertionError("project schema imported")\n'
    )
    (target / f"{module}.py").write_text(
        "import schema\n"
        "from bentoml.legacy import Service\n"
        "svc = Service('target_project')\n"
        "schema_class = schema.Schema\n"
    )
    svc = import_service(f"{module}:svc", working_dir=str(target))

    assert svc.name == "target_project"
    assert sys.modules[module].schema_class is schema.Schema


def test_working_dir_priority_preserves_successful_model_store(
    conflicting_service_projects, tmp_path
):
    from bentoml._internal.configuration.containers import BentoMLContainer
    from bentoml._internal.models import ModelStore

    module, (_, target) = conflicting_service_projects
    original_store = BentoMLContainer.model_store.get()
    model_store = ModelStore(str(tmp_path / "successful_store"))
    try:
        svc = import_service(
            f"{module}:svc", working_dir=str(target), model_store=model_store
        )
        assert svc.name == "target_project"
        assert BentoMLContainer.model_store.get() is model_store
    finally:
        BentoMLContainer.model_store.set(original_store)


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
        ImportServiceError, match="Multiple `bentoml.Service` instances found in module"
    ):
        import_service("bento_service_in_module.multi_service")

    svc = import_service("multi_service_in_package:svc_i")
    assert svc.name == "test-multi-service-in-package-i"

    svc = import_service("multi_service_in_package:svc_ii")
    assert svc.name == "test-multi-service-in-package-ii"

    with pytest.raises(
        ImportServiceError, match="Multiple `bentoml.Service` instances found in module"
    ):
        import_service("multi_service_in_package")
