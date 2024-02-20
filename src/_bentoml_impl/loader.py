from __future__ import annotations

import importlib
import pathlib
import sys
import typing as t

import yaml

if t.TYPE_CHECKING:
    from _bentoml_sdk import Service

BENTO_YAML_FILENAME = "bento.yaml"


def normalize_identifier(
    service_identifier: str,
    working_dir: str | None = None,
) -> tuple[str, pathlib.Path]:
    """
    Normalize a service identifier to a package:Service format, and return the bento
    path.

    valid identifiers:
    - package:Service                        # bentoml serve projects or normalized
    - package:Service.dependency             # bentocloud dependencies services
    - ~/bentoml/bentos/my_bento/version      # bentocloud entry service
    - ~/bentoml/bentos/my_bento/version/bento.yaml
    - bento1/bentofile.yaml                  # bentoml serve from multi-target projects
    - my_service:a7ab819                     # bentoml serve built bentos
    - package.py:Service
    - package.py:Service.dependency
    - .
    """
    if working_dir is not None:
        path = pathlib.Path(working_dir).joinpath(service_identifier)
    else:
        path = pathlib.Path(service_identifier)
    if path.exists():
        if path.is_file() and path.name == BENTO_YAML_FILENAME:
            # this is a bento.yaml file
            yaml_path = path
            bento_path = path.parent.joinpath("src")
        elif path.is_dir() and path.joinpath(BENTO_YAML_FILENAME).is_file():
            # this is a bento directory
            yaml_path = path.joinpath(BENTO_YAML_FILENAME)
            bento_path = path.joinpath("src")
        elif path.is_file() and path.name == "bentofile.yaml":
            # this is a bentofile.yaml file
            yaml_path = path
            bento_path = (
                pathlib.Path(working_dir) if working_dir is not None else path.parent
            )
        elif path.is_dir() and path.joinpath("bentofile.yaml").is_file():
            # this is a bento project directory
            yaml_path = path.joinpath("bentofile.yaml")
            bento_path = (
                pathlib.Path(working_dir) if working_dir is not None else path.parent
            )
        else:
            raise ValueError(f"found a path but not a bento: {service_identifier}")

        with open(yaml_path, "r") as f:
            bento_yaml = yaml.safe_load(f)
        assert "service" in bento_yaml, "service field is required in bento.yaml"
        return normalize_package(bento_yaml["service"]), bento_path

    elif ":" in service_identifier:
        # a python import str or a built bento in store

        # TODO(jiang): bento store configs are sdk configs, should be moved to sdk in the future
        from bentoml._internal.configuration.containers import BentoMLContainer
        from bentoml.exceptions import NotFound

        bento_store = BentoMLContainer.bento_store.get()

        try:
            bento = bento_store.get(service_identifier)
        except NotFound:
            # a python import str
            return normalize_package(service_identifier), pathlib.Path(
                working_dir if working_dir is not None else "."
            )
        else:
            # a built bento in bento store

            yaml_path = pathlib.Path(bento.path).joinpath(BENTO_YAML_FILENAME)
            with open(yaml_path, "r") as f:
                bento_yaml = yaml.safe_load(f)
            assert "service" in bento_yaml, "service field is required in bento.yaml"
            return normalize_package(bento_yaml["service"]), yaml_path.parent.joinpath(
                "src"
            )
    else:
        raise ValueError(f"invalid service: {service_identifier}")


def import_service(
    service_identifier: str,
    bento_path: pathlib.Path | None = None,
) -> Service[t.Any]:
    """
    import a service from a service identifier, which should be normalized by
    `normalize_identifier` function.
    """
    from _bentoml_sdk import Service

    if bento_path is None:
        bento_path = pathlib.Path(".").absolute()

    # patch python path if needed
    if bento_path != pathlib.Path("."):
        # a project
        extra_python_path = str(bento_path.absolute())
        sys.path.insert(0, extra_python_path)
    else:
        # a project under current directory
        extra_python_path = None

    # patch model store if needed
    if (
        bento_path.parent.joinpath(BENTO_YAML_FILENAME).exists()
        and bento_path.parent.joinpath("models").exists()
    ):
        from bentoml._internal.configuration.containers import BentoMLContainer
        from bentoml._internal.models import ModelStore

        original_model_store = BentoMLContainer.model_store.get()

        BentoMLContainer.model_store.set(
            ModelStore((bento_path.parent.joinpath("models").absolute()))
        )
    else:
        original_model_store = None

    try:
        module_name, _, attrs_str = service_identifier.partition(":")

        assert (
            module_name and attrs_str
        ), f'Invalid import target "{service_identifier}", must format as "<module>:<attribute>"'

        module = importlib.import_module(module_name)
        root_service_name, _, depend_path = attrs_str.partition(".")
        root_service = getattr(module, root_service_name)

        assert isinstance(
            root_service, Service
        ), f'import target "{module_name}:{attrs_str}" is not a bentoml.Service instance'

        if not depend_path:
            return root_service  # type: ignore
        else:
            return root_service.find_dependent(depend_path)

    except (ImportError, AttributeError, KeyError, AssertionError) as e:
        sys_path = sys.path.copy()
        if extra_python_path is not None:
            sys.path.remove(extra_python_path)

        if original_model_store is not None:
            from bentoml._internal.configuration.containers import BentoMLContainer

            BentoMLContainer.model_store.set(original_model_store)
        from bentoml.exceptions import ImportServiceError

        raise ImportServiceError(
            f'Failed to import service "{service_identifier}": {e}, sys.path: {sys_path}, cwd: {pathlib.Path.cwd()}'
        ) from None


def normalize_package(service_identifier: str) -> str:
    """
    service.py:Service -> service:Service
    """
    package, _, service_path = service_identifier.partition(":")
    if package.endswith(".py"):
        package = package[:-3]
    return ":".join([package, service_path])
