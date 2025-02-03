from __future__ import annotations

import contextlib
import importlib
import logging
import os
import pathlib
import sys
import typing as t

import yaml

from bentoml._internal.bento.bento import BENTO_YAML_FILENAME
from bentoml._internal.bento.bento import DEFAULT_BENTO_BUILD_FILES
from bentoml._internal.context import server_context

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


if t.TYPE_CHECKING:
    from _bentoml_sdk import Service


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
        elif path.is_file() and path.name in DEFAULT_BENTO_BUILD_FILES:
            # this is a bento build config file
            yaml_path = path
            bento_path = (
                pathlib.Path(working_dir) if working_dir is not None else path.parent
            )
        elif path.is_dir() and any(
            path.joinpath(filename).is_file() for filename in DEFAULT_BENTO_BUILD_FILES
        ):
            # this is a bento project directory
            yaml_path = next(
                path.joinpath(filename)
                for filename in DEFAULT_BENTO_BUILD_FILES
                if path.joinpath(filename).is_file()
            )
            bento_path = pathlib.Path(working_dir) if working_dir is not None else path
        else:
            raise ValueError(f"found a path but not a bento: {service_identifier}")

        if yaml_path.name == "pyproject.toml":
            with yaml_path.open("rb") as f:
                data = tomllib.load(f)
            bento_yaml = data.get("tool", {}).get("bentoml", {}).get("build", {})
        else:
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
            return normalize_package(bento_yaml["service"]), pathlib.Path(
                bento.path_of("src")
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
    from bentoml._internal.bento.bento import Bento
    from bentoml._internal.bento.build_config import BentoBuildConfig
    from bentoml._internal.configuration.containers import BentoMLContainer

    if bento_path is None:
        bento_path = pathlib.Path(".")
    bento_path = bento_path.absolute()

    # patch python path if needed
    if bento_path != pathlib.Path(".").absolute():
        # a project
        extra_python_path = str(bento_path)
        sys.path.insert(0, extra_python_path)
    else:
        # a project under current directory
        extra_python_path = None

    # patch model store if needed
    if (
        bento_path.with_name(BENTO_YAML_FILENAME).exists()
        and bento_path.with_name("models").exists()
    ):
        from bentoml._internal.models import ModelStore

        original_path = os.getcwd()
        original_model_store = BentoMLContainer.model_store.get()
        # cwd into this for relative path to work
        os.chdir(bento_path.absolute())

        # check in bento source

        BentoMLContainer.model_store.set(
            ModelStore((bento_path.with_name("models").absolute()))
        )
    else:
        original_path = None
        original_model_store = None

    # load model aliases
    bento: Bento | None = None
    if bento_path.with_name(BENTO_YAML_FILENAME).exists():
        bento = Bento.from_path(str(bento_path.parent))
        model_aliases = {m.alias: str(m.tag) for m in bento.info.all_models if m.alias}
    else:
        model_aliases = {}
        for filename in DEFAULT_BENTO_BUILD_FILES:
            if (bentofile := bento_path.joinpath(filename)).exists():
                build_config = BentoBuildConfig.from_file(bentofile)
                model_aliases = build_config.model_aliases
                break
    BentoMLContainer.model_aliases.set(model_aliases)

    try:
        module_name, _, attrs_str = service_identifier.partition(":")

        assert module_name and attrs_str, (
            f'Invalid import target "{service_identifier}", must format as "<module>:<attribute>"'
        )
        module = importlib.import_module(module_name)
        root_service_name, _, depend_path = attrs_str.partition(".")
        root_service = t.cast("Service[t.Any]", getattr(module, root_service_name))

        assert isinstance(root_service, Service), (
            f'import target "{module_name}:{attrs_str}" is not a bentoml.Service instance'
        )

        if not depend_path:
            svc = root_service
        else:
            svc = root_service.find_dependent_by_path(depend_path)
        if bento is not None:
            svc.on_load_bento(bento)
        return svc

    except (ImportError, AttributeError, KeyError, AssertionError) as e:
        sys_path = sys.path.copy()
        if extra_python_path is not None:
            sys.path.remove(extra_python_path)

        if original_model_store is not None:
            from bentoml._internal.configuration.containers import BentoMLContainer

            BentoMLContainer.model_store.set(original_model_store)

        if original_path is not None:
            os.chdir(original_path)

        from bentoml.exceptions import ImportServiceError

        message = f'Failed to import service "{service_identifier}": {e}, sys.path: {sys_path}, cwd: {pathlib.Path.cwd()}'
        if (
            isinstance(e, ImportError)
            and server_context.worker_index is None
            and not (e.name or "").startswith(("bentoml", "_bentoml_"))
        ):
            message += "\nIf you are trying to import a runtime-only module, try wrapping it inside `with bentoml.importing():`"

        raise ImportServiceError(message) from None


def normalize_package(service_identifier: str) -> str:
    """
    service.py:Service -> service:Service
    """
    package, _, service_path = service_identifier.partition(":")
    if package.endswith(".py"):
        package = package[:-3]
    return ":".join([package, service_path])


@contextlib.contextmanager
def importing() -> t.Iterator[None]:
    """A context manager that suppresses the ImportError when importing runtime-specific modules.

    Example:

    .. code-block:: python

        with bentoml.importing():
            import torch
            import tensorflow
    """

    logger = logging.getLogger("bentoml.service")

    try:
        yield
    except ImportError as e:
        if server_context.worker_index is not None:
            raise
        logger.info(
            f"Skipping import of module {e.name} because it is not available in the current environment"
        )
