import os
import sys
import typing as t
import logging
import importlib
from typing import TYPE_CHECKING

import fs
from simple_di import inject
from simple_di import Provide

from ..bento import Bento
from ..models import ModelStore
from ...exceptions import NotFound
from ...exceptions import BentoMLException
from ..bento.bento import BENTO_YAML_FILENAME
from ..bento.bento import BENTO_PROJECT_DIR_NAME
from ..configuration import BENTOML_VERSION
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..bento import BentoStore
    from .service import Service

logger = logging.getLogger(__name__)


class ImportServiceError(BentoMLException):
    pass


@inject
def import_service(
    svc_import_path: str,
    *,
    working_dir: t.Optional[str] = None,
    change_global_cwd: bool = False,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "Service":
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

    prev_cwd = None
    sys_path_modified = False

    try:
        if working_dir is not None:
            working_dir = os.path.realpath(os.path.expanduser(working_dir))
            if change_global_cwd:
                # Set cwd(current working directory) to the Bento's project directory,
                # which allows user code to read files using relative path
                prev_cwd = os.getcwd()
                os.chdir(working_dir)
        else:
            working_dir = os.getcwd()

        if working_dir not in sys.path:
            sys.path.insert(0, working_dir)
            sys_path_modified = True

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

        if os.path.exists(import_path):
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
            module_name_parts = []
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
        BentoMLContainer.model_store.set(model_store)
        try:
            module = importlib.import_module(module_name, package=working_dir)
        except ImportError as e:
            raise ImportServiceError(
                f'{e} happens when importing "{module_name}" in '
                f'current path: {repr(sys.path)}. working dir: "{working_dir}", '
                f'current dir: "{os.getcwd()}"'
            )
        finally:
            # Reset to default local model store
            BentoMLContainer.model_store.reset()

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
                (k, v) for k, v in module.__dict__.items() if isinstance(v, Service)
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
            instance, Service
        ), f'import target "{module_name}:{attrs_str}" is not a bentoml.Service instance'
        instance._working_dir = working_dir  # type: ignore
        instance._import_str = f"{module_name}:{attrs_str}"  # type: ignore
        return instance
    except ImportServiceError:
        if prev_cwd and change_global_cwd:
            # Reset to previous cwd
            os.chdir(prev_cwd)
        if sys_path_modified and working_dir:
            # Undo changes to sys.path
            sys.path.remove(working_dir)
        raise


@inject
def load_bento(
    bento_tag: str,
    bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
    change_global_cwd: bool = False,
) -> "Service":
    """Load a Service instance from a bento found in local bento store:

    Example usage:
        load_bento("FraudDetector:latest")
        load_bento("FraudDetector:20210709_DE14C9")
    """
    bento = bento_store.get(bento_tag)
    logger.debug(
        'Loading bento "%s" found in local store: %s',
        bento.tag,
        bento._fs.getsyspath("/"),
    )

    # not in validate as it's only really necessary when getting bentos from disk
    if bento.info.bentoml_version != BENTOML_VERSION:
        logger.warning(
            f'Bento "{bento.tag}" was built with BentoML version {bento.info.bentoml_version}, which does not match the current BentoML version {BENTOML_VERSION}'
        )
    return _load_bento(bento, change_global_cwd)


def load_bento_dir(path: str, change_global_cwd: bool = False) -> "Service":
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
    return _load_bento(bento, change_global_cwd)


def _load_bento(bento: Bento, change_global_cwd: bool) -> "Service":
    # Use Bento's user project path as working directory when importing the service
    working_dir = bento._fs.getsyspath(BENTO_PROJECT_DIR_NAME)

    # Use Bento's local "{base_dir}/models/" directory as its model store
    model_store = ModelStore(bento._fs.getsyspath("models"))

    svc = import_service(
        bento.info.service,
        working_dir=working_dir,
        change_global_cwd=change_global_cwd,
        model_store=model_store,
    )
    svc.version = bento.info.tag.version
    svc.tag = bento.info.tag
    svc.bento = bento
    return svc


def load(
    bento_identifier: str,
    working_dir: t.Optional[str] = None,
    change_global_cwd: bool = False,
) -> "Service":
    """Load a Service instance by the bento_identifier

    A bento_identifier:str can be provided in three different forms:

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
    if os.path.isdir(os.path.expanduser(bento_identifier)) and os.path.isfile(
        os.path.expanduser(os.path.join(bento_identifier, BENTO_YAML_FILENAME))
    ):
        try:
            svc = load_bento_dir(bento_identifier, change_global_cwd=change_global_cwd)
        except ImportServiceError as e:
            raise BentoMLException(
                f"Failed loading Bento from directory {bento_identifier}: {e}"
            )
        logger.info("Service loaded from Bento directory: %s", svc)
    else:
        try:
            svc = import_service(
                bento_identifier,
                working_dir=working_dir,
                change_global_cwd=change_global_cwd,
            )
            logger.info("Service imported from source: %s", svc)
        except ImportServiceError as e1:
            try:
                svc = load_bento(bento_identifier, change_global_cwd=change_global_cwd)
                logger.info("Service loaded from Bento store: %s", svc)
            except (NotFound, ImportServiceError) as e2:
                raise BentoMLException(
                    f"Failed to load bento or import service "
                    f"'{bento_identifier}'. If you are attempting to "
                    f"import bento in local store: `{e1}`, or if you are importing by "
                    f"python module path: `{e2}`"
                )
    return svc
