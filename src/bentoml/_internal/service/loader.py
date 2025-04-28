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

from ...exceptions import ImportServiceError
from ...exceptions import NotFound
from ..bento import Bento
from ..bento.bento import BENTO_PROJECT_DIR_NAME
from ..bento.bento import BENTO_YAML_FILENAME
from ..bento.bento import DEFAULT_BENTO_BUILD_FILES
from ..bento.build_config import BentoBuildConfig
from ..configuration import BENTOML_VERSION
from ..configuration.containers import BentoMLContainer
from ..models import ModelStore
from ..tag import Tag
from ..utils.args import set_arguments
from ..utils.uri import encode_path_for_uri

if TYPE_CHECKING:
    from _bentoml_sdk import Service as NewService

    from ..bento import BentoStore
    from .service import Service

    AnyService = t.Union[Service, NewService[t.Any]]

logger = logging.getLogger(__name__)


@inject
def import_service(
    svc_import_path: str,
    *,
    working_dir: t.Optional[str] = None,
    reload: bool = False,
    standalone_load: bool = False,
    model_store: ModelStore = Provide[BentoMLContainer.model_store],
) -> AnyService:
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

    sys_path_modified = False
    prev_cwd = None
    global_model_store = BentoMLContainer.model_store.get()

    if working_dir is not None:
        prev_cwd = os.getcwd()
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

    def _restore(restore_model_store: bool = False) -> None:
        """Undo changes to sys.path, cwd and model store"""
        if restore_model_store and model_store is not global_model_store:
            BentoMLContainer.model_store.set(global_model_store)
        if sys_path_modified and working_dir:
            sys.path.remove(working_dir)
        if prev_cwd is not None:
            os.chdir(prev_cwd)

    try:
        svc = _do_import(svc_import_path, working_dir, reload)
    except ImportServiceError:
        _restore(True)
        raise
    else:
        if not standalone_load:
            _restore()
        return svc


def _do_import(
    svc_import_path: str, working_dir: str, reload: bool = False
) -> AnyService:
    from .service import Service

    try:
        from _bentoml_sdk import Service as NewService

    except ImportError:

        class NewService:
            pass

    logger.debug(
        'Importing service "%s" from working dir: "%s"', svc_import_path, working_dir
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
    needs_reload = module_name in sys.modules
    try:
        module = importlib.import_module(module_name, package=working_dir)
    except ImportError as e:
        from ..context import server_context

        message = f'Failed to import module "{module_name}": {e}'
        if server_context.worker_index is None and not (e.name or "").startswith(
            ("bentoml", "_bentoml_")
        ):
            message += "\nIf you are trying to import a runtime-only module, try wrapping it inside `with bentoml.importing():`"
        raise ImportServiceError(message)
    if reload and needs_reload:
        importlib.reload(module)

    if attrs_str:
        instance = module
        try:
            for attr_str in attrs_str.split("."):
                if isinstance(instance, NewService):
                    instance = instance.dependencies[attr_str].on
                else:
                    instance = getattr(instance, attr_str)
        except (AttributeError, KeyError):
            raise ImportServiceError(
                f'Attribute "{attrs_str}" not found in module "{module_name}".'
            )
    else:
        instances = [
            (k, v) for k, v in module.__dict__.items() if isinstance(v, NewService)
        ]
        dependents: set[NewService[t.Any]] = set()
        for _, svc in instances:
            for dep in svc.dependencies.values():
                if dep.on is not None:
                    dependents.add(dep.on)
        instances = [(k, v) for k, v in instances if v not in dependents]
        if not instances:  # Fallback to 1.1 service
            instances = [
                (k, v) for k, v in module.__dict__.items() if isinstance(v, Service)
            ]

        if not instances:
            raise ImportServiceError(f"No service found in the module '{module_name}'")
        elif len(instances) == 1:
            attrs_str, instance = instances[0]
        else:
            raise ImportServiceError(
                f'Multiple Service instances found in module "{module_name}", use '
                '"<module>:<svc_variable_name>" to specify the service instance or '
                "define only one service instance per python module/file"
            )

    # set import_str for retrieving the service import origin
    object.__setattr__(instance, "_import_str", f"{module_name}:{attrs_str}")
    return t.cast("AnyService", instance)


@inject
def load_bento(
    bento: str | Tag | Bento,
    reload: bool = False,
    standalone_build: bool = False,
    bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> AnyService:
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
    return _load_bento(bento, reload=reload, standalone_build=standalone_build)


def load_bento_dir(
    path: str, reload: bool = False, standalone_build: bool = False
) -> AnyService:
    """Load a Service instance from a bento directory

    Example usage:
        load_bento_dir("~/bentoml/bentos/iris_classifier/4tht2icroji6zput3suqi5nl2")
    """
    bento_fs = fs.open_fs(encode_path_for_uri(path))
    bento = Bento.from_fs(bento_fs)
    logger.debug(
        'Loading bento "%s" from directory: %s',
        bento.tag,
        path,
    )
    return _load_bento(bento, reload=reload, standalone_build=standalone_build)


def _load_bento(
    bento: Bento, reload: bool = False, standalone_build: bool = False
) -> AnyService:
    # Use Bento's user project path as working directory when importing the service
    set_arguments(**bento.info.args)
    working_dir = bento.path_of(BENTO_PROJECT_DIR_NAME)

    model_store = BentoMLContainer.model_store.get()
    # read from bento's local model store if it exists and is not empty
    # This is the case when running in a container
    local_model_store = bento._model_store
    if local_model_store is not None and len(local_model_store.list()) > 0:
        model_store = local_model_store

    # Read the model aliases
    resolved_model_aliases = {
        m.alias: str(m.tag) for m in bento.info.all_models if m.alias
    }
    BentoMLContainer.model_aliases.set(resolved_model_aliases)

    svc = import_service(
        bento.info.service,
        working_dir=working_dir,
        reload=reload,
        standalone_load=standalone_build,
        model_store=model_store,
    )
    svc.on_load_bento(bento)
    return svc


def load(
    bento_identifier: str | Tag | Bento,
    working_dir: t.Optional[str] = None,
    reload: bool = False,
    standalone_load: bool = False,
) -> AnyService:
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
    """
    if isinstance(bento_identifier, (Bento, Tag)):
        # Load from local BentoStore
        return load_bento(
            bento_identifier, reload=reload, standalone_build=standalone_load
        )
    bento_path = os.path.expanduser(
        os.path.abspath(os.path.join(working_dir or ".", bento_identifier))
    )
    if os.path.isdir(bento_path):
        if os.path.isfile(
            os.path.expanduser(os.path.join(bento_path, BENTO_YAML_FILENAME))
        ):
            # Loading from path to a built Bento
            svc = load_bento_dir(
                bento_path, reload=reload, standalone_build=standalone_load
            )
            logger.info("Service loaded from Bento directory: %s", svc)
        else:
            for filename in DEFAULT_BENTO_BUILD_FILES:
                if os.path.isfile(
                    config_file := os.path.expanduser(
                        os.path.join(bento_path, filename)
                    )
                ):
                    build_config = BentoBuildConfig.from_file(config_file)
                    BentoMLContainer.model_aliases.set(build_config.model_aliases)
                    if build_config.service:
                        svc = import_service(
                            build_config.service,
                            working_dir=bento_path,
                            reload=reload,
                            standalone_load=standalone_load,
                        )
                        logger.debug(
                            "'%s' loaded from '%s': %s", svc.name, bento_path, svc
                        )
                        break
            else:
                if os.path.isfile(os.path.join(bento_path, "service.py")):
                    logger.info("Loading service from default location 'service.py'")
                    svc = import_service(
                        "service.py",
                        working_dir=bento_path,
                        reload=reload,
                        standalone_load=standalone_load,
                    )
                else:
                    raise ImportServiceError(
                        f"Failed to load bento or import service '{bento_identifier}'. "
                        f"The directory '{bento_path}' does not contain a valid bentofile.yaml or service.py."
                    )
    else:
        try:
            # Loading from service definition file, e.g. "my_service.py:svc"
            svc = import_service(
                bento_identifier,
                working_dir=working_dir,
                reload=reload,
                standalone_load=standalone_load,
            )
            logger.debug("'%s' imported from source: %s", svc.name, svc)
        except ImportServiceError as e1:
            try:
                # Loading from local bento store by tag, e.g. "iris_classifier:latest"
                svc = load_bento(
                    bento_identifier, reload=reload, standalone_build=standalone_load
                )
                logger.debug("'%s' loaded from Bento store: %s", svc.name, svc)
            except (NotFound, ImportServiceError) as e2:
                raise ImportServiceError(
                    f"Failed to load bento or import service '{bento_identifier}'.\n"
                    f"If you are attempting to import bento by tag: '{e2}'.\n"
                    f"If you are importing by python module path: '{e1}'."
                ) from e2
    return svc
