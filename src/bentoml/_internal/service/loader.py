from __future__ import annotations

import importlib
import logging
import os
import sys
import typing as t
from typing import TYPE_CHECKING

import fs
from simple_di import Provide
from simple_di import inject

from ...exceptions import BentoMLException
from ...exceptions import ImportServiceError
from ...exceptions import NotFound
from ..bento import Bento
from ..bento.bento import BENTO_PROJECT_DIR_NAME
from ..bento.bento import BENTO_YAML_FILENAME
from ..bento.bento import DEFAULT_BENTO_BUILD_FILE
from ..bento.build_config import BentoBuildConfig
from ..configuration import BENTOML_VERSION
from ..configuration.containers import BentoMLContainer
from ..models import ModelStore
from ..tag import Tag
from .service import on_load_bento

if TYPE_CHECKING:
    from _bentoml_sdk import Service as NewService

    from ..bento import BentoStore
    from .service import Service

logger = logging.getLogger(__name__)


@inject
def import_service(
    svc_import_path: str,
    *,
    working_dir: t.Optional[str] = None,
    standalone_load: bool = False,
    model_store: ModelStore = Provide[BentoMLContainer.model_store],
) -> Service | NewService[t.Any]:
    """Import a Service instance from source code, by providing the svc_import_path
    which represents the module where the Service instance is created and optionally
    what attribute can be used to access this Service instance in that module

    Example usage:
        # When multiple service defined in the same module
        import_service("fraud_detector:svc_a")
        import_service("fraud_detector:svc_b")

        # Find svc by Python module name or file path
        import_service("fraud_detector:svc")
        import_service("fraud_detector.py:svc")
        import_service("foo.bar.fraud_detector:svc")
        import_service("./def/abc/fraud_detector.py:svc")

        # When there's only one Service instance in the target module, the attributes
        # part in the svc_import_path can be omitted
        import_service("fraud_detector.py")
        import_service("fraud_detector")
    """
    from bentoml import Service

    service_types: list[type] = [Service]
    try:
        from _bentoml_sdk import Service as NewService

        service_types.append(NewService)
    except ImportError:
        pass

    prev_cwd = None
    sys_path_modified = False
    prev_cwd = os.getcwd()
    global_model_store = BentoMLContainer.model_store.get()

    def recover_standalone_env_change():
        # Reset to previous cwd
        os.chdir(prev_cwd)
        BentoMLContainer.model_store.set(global_model_store)

    try:
        if working_dir is not None:
            working_dir = os.path.realpath(os.path.expanduser(working_dir))
            # Set cwd(current working directory) to the Bento's project directory,
            # which allows user code to read files using relative path
            os.chdir(working_dir)
        else:
            working_dir = os.getcwd()

        if working_dir not in sys.path:
            sys.path.insert(0, working_dir)
            sys_path_modified = True

        if model_store is not global_model_store:
            BentoMLContainer.model_store.set(model_store)

        logger.debug(
            'Importing service "%s" from working dir: "%s"',
            svc_import_path,
            working_dir,
        )

        import_path, _, attrs_str = svc_import_path.partition(":")
        if not import_path:
            raise ImportServiceError(
                f'Invalid import target "{svc_import_path}", must format as '
                '"<module>:<attribute>" or "<module>'
            )

        if os.path.isfile(import_path):
            import_path = os.path.realpath(import_path)
            # Importing from a module file path:
            if not import_path.startswith(working_dir):
                raise ImportServiceError(
                    f'Module "{import_path}" not found in working directory "{working_dir}"'
                )

            file_name, ext = os.path.splitext(import_path)
            if ext != ".py":
                raise ImportServiceError(
                    f'Invalid module extension "{ext}" in target "{svc_import_path}",'
                    ' the only extension acceptable here is ".py"'
                )

            # move up until no longer in a python package or in the working dir
            module_name_parts: t.List[str] = []
            path = file_name
            while True:
                path, name = os.path.split(path)
                module_name_parts.append(name)
                if (
                    not os.path.exists(os.path.join(path, "__init__.py"))
                    or path == working_dir
                ):
                    break
            module_name = ".".join(module_name_parts[::-1])
        else:
            # Importing by module name:
            module_name = import_path

        # Import the service using the Bento's own model store
        try:
            module = importlib.import_module(module_name, package=working_dir)
        except ImportError as e:
            raise ImportServiceError(f'Failed to import module "{module_name}": {e}')
        if not standalone_load:
            recover_standalone_env_change()

        if attrs_str:
            instance = module
            try:
                for attr_str in attrs_str.split("."):
                    instance = getattr(instance, attr_str)
            except AttributeError:
                raise ImportServiceError(
                    f'Attribute "{attrs_str}" not found in module "{module_name}".'
                )
        else:
            instances = [
                (k, v)
                for k, v in module.__dict__.items()
                if isinstance(v, tuple(service_types))
            ]

            if len(instances) == 1:
                attrs_str = instances[0][0]
                instance = instances[0][1]
            else:
                raise ImportServiceError(
                    f'Multiple Service instances found in module "{module_name}", use'
                    '"<module>:<svc_variable_name>" to specify the service instance or'
                    "define only service instance per python module/file"
                )

        assert isinstance(
            instance, tuple(service_types)
        ), f'import target "{module_name}:{attrs_str}" is not a bentoml.Service instance'

        # set import_str for retrieving the service import origin
        object.__setattr__(instance, "_import_str", f"{module_name}:{attrs_str}")
        return t.cast("Service | NewService[t.Any]", instance)
    except ImportServiceError:
        if sys_path_modified and working_dir:
            # Undo changes to sys.path
            sys.path.remove(working_dir)

        recover_standalone_env_change()
        raise


@inject
def load_bento(
    bento: str | Tag | Bento,
    bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
    standalone_load: bool = False,
) -> Service | NewService[t.Any]:
    """Load a Service instance from a bento found in local bento store:

    Example usage:
        load_bento("FraudDetector:latest")
        load_bento("FraudDetector:20210709_DE14C9")
    """
    if isinstance(bento, (str, Tag)):
        bento = bento_store.get(bento)

    logger.debug(
        'Loading bento "%s" found in local store: %s',
        bento.tag,
        bento.path,
    )

    # not in validate as it's only really necessary when getting bentos from disk
    if bento.info.bentoml_version != BENTOML_VERSION:
        info_bentoml_version = bento.info.bentoml_version
        if tuple(info_bentoml_version.split(".")) > tuple(BENTOML_VERSION.split(".")):
            logger.warning(
                "%s was built with newer version of BentoML, which does not match with current running BentoML version %s",
                bento,
                BENTOML_VERSION,
            )
        else:
            logger.debug(
                "%s was built with BentoML version %s, which does not match the current BentoML version %s",
                bento,
                info_bentoml_version,
                BENTOML_VERSION,
            )
    return _load_bento(bento, standalone_load)


def load_bento_dir(
    path: str, standalone_load: bool = False
) -> Service | NewService[t.Any]:
    """Load a Service instance from a bento directory

    Example usage:
        load_bento_dir("~/bentoml/bentos/iris_classifier/4tht2icroji6zput3suqi5nl2")
    """
    bento_fs = fs.open_fs(path)
    bento = Bento.from_fs(bento_fs)
    logger.debug(
        'Loading bento "%s" from directory: %s',
        bento.tag,
        path,
    )
    return _load_bento(bento, standalone_load)


def _load_bento(bento: Bento, standalone_load: bool) -> Service | NewService[t.Any]:
    # Use Bento's user project path as working directory when importing the service
    working_dir = bento._fs.getsyspath(BENTO_PROJECT_DIR_NAME)

    model_store = BentoMLContainer.model_store.get()
    # read from bento's local model store if it exists and is not empty
    # This is the case when running in a container
    local_model_store = bento._model_store
    if local_model_store is not None and len(local_model_store.list()) > 0:
        model_store = local_model_store

    # Read the model aliases
    resolved_model_aliases = {m.alias: str(m.tag) for m in bento.info.models if m.alias}
    BentoMLContainer.model_aliases.set(resolved_model_aliases)

    svc = import_service(
        bento.info.service,
        working_dir=working_dir,
        standalone_load=standalone_load,
        model_store=model_store,
    )
    on_load_bento(svc, bento)
    return svc


def load(
    bento_identifier: str | Tag | Bento,
    working_dir: t.Optional[str] = None,
    standalone_load: bool = False,
) -> Service | NewService[t.Any]:
    """Load a Service instance by the bento_identifier

    Args:
        bento_identifier: target Service to import or Bento to load
        working_dir: when importing from service, set the working_dir
        standalone_load: treat target Service as standalone. This will change global
                         current working directory and global model store.

    Returns:
        The loaded :obj:`bentoml.Service` instance.

    The argument ``bento_identifier`` can be one of the following forms:

    * Tag pointing to a Bento in local Bento store under `BENTOML_HOME/bentos`
    * File path to a Bento directory
    * "import_str" for loading a service instance from the `working_dir`

    Example load from Bento usage:

    .. code-block:: python

        # load from local bento store
        load("FraudDetector:latest")
        load("FraudDetector:4tht2icroji6zput")

        # load from bento directory
        load("~/bentoml/bentos/iris_classifier/4tht2icroji6zput")


    Example load from working directory by "import_str" usage:

    .. code-block:: python

        # When multiple service defined in the same module
        load("fraud_detector:svc_a")
        load("fraud_detector:svc_b")

        # Find svc by Python module name or file path
        load("fraud_detector:svc")
        load("fraud_detector.py:svc")
        load("foo.bar.fraud_detector:svc")
        load("./def/abc/fraud_detector.py:svc")

        # When there's only one Service instance in the target module, the attributes
        # part in the svc_import_path can be omitted
        load("fraud_detector.py")
        load("fraud_detector")

    Limitations when `standalone_load=False`:

        * Models used in the Service being imported, if not accessed during
          module import, must be presented in the global model store
        * Files required for the Service to run, if not accessed during module
          import, must be presented in the current working directory
    """
    if isinstance(bento_identifier, (Bento, Tag)):
        # Load from local BentoStore
        return load_bento(bento_identifier)

    if os.path.isdir(os.path.expanduser(bento_identifier)):
        bento_path = os.path.abspath(os.path.expanduser(bento_identifier))

        if os.path.isfile(
            os.path.expanduser(os.path.join(bento_path, BENTO_YAML_FILENAME))
        ):
            # Loading from path to a built Bento
            try:
                svc = load_bento_dir(bento_path, standalone_load=standalone_load)
            except ImportServiceError as e:
                raise BentoMLException(
                    f"Failed loading Bento from directory {bento_path}: {e}"
                )
            logger.info("Service loaded from Bento directory: %s", svc)
        elif os.path.isfile(
            os.path.expanduser(os.path.join(bento_path, DEFAULT_BENTO_BUILD_FILE))
        ):
            # Loading from path to a project directory containing bentofile.yaml
            try:
                with open(
                    os.path.join(bento_path, DEFAULT_BENTO_BUILD_FILE),
                    "r",
                    encoding="utf-8",
                ) as f:
                    build_config = BentoBuildConfig.from_yaml(f)
                assert (
                    build_config.service
                ), '"service" field in "bentofile.yaml" is required for loading the service, e.g. "service: my_service.py:svc"'
                BentoMLContainer.model_aliases.set(build_config.model_aliases)
                svc = import_service(
                    build_config.service,
                    working_dir=bento_path,
                    standalone_load=standalone_load,
                )
            except ImportServiceError as e:
                raise BentoMLException(
                    f"Failed loading Bento from directory {bento_path}: {e}"
                )
            logger.debug("'%s' loaded from '%s': %s", svc.name, bento_path, svc)
        else:
            raise BentoMLException(
                f"Failed loading service from path {bento_path}. When loading from a path, it must be either a Bento containing bento.yaml or a project directory containing bentofile.yaml"
            )
    else:
        try:
            # Loading from service definition file, e.g. "my_service.py:svc"
            svc = import_service(
                bento_identifier,
                working_dir=working_dir,
                standalone_load=standalone_load,
            )
            logger.debug("'%s' imported from source: %s", svc.name, svc)
        except ImportServiceError as e1:
            try:
                # Loading from local bento store by tag, e.g. "iris_classifier:latest"
                svc = load_bento(bento_identifier, standalone_load=standalone_load)
                logger.debug("'%s' loaded from Bento store: %s", svc.name, svc)
            except (NotFound, ImportServiceError) as e2:
                raise BentoMLException(
                    f"Failed to load bento or import service '{bento_identifier}'.\n"
                    f"If you are attempting to import bento in local store: '{e1}'.\n"
                    f"If you are importing by python module path: '{e2}'."
                )
    return svc
