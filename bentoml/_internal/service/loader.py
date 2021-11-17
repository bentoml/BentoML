import importlib
import logging
import os
import sys
import typing as t
from typing import TYPE_CHECKING

from simple_di import Provide, inject

from ...exceptions import BentoMLException, NotFound
from ..bento.bento import BENTO_PROJECT_DIR_NAME
from ..configuration.containers import BentoMLContainer
from ..models.store import ModelStore

if TYPE_CHECKING:  # pragma: no cover
    from ..bento import BentoStore
    from .service import Service

logger = logging.getLogger(__name__)


class ImportServiceError(BentoMLException):
    pass


@inject
def import_service(
    svc_import_path: str,
    working_dir: t.Optional[str] = None,
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
            working_dir = os.path.realpath(working_dir)
            # Set cwd(current working directory) to the Bento's project directory, which
            # allows user code to read files using relative path
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

        default_model_store = BentoMLContainer.model_store
        # Import the service using the Bento's own model store
        BentoMLContainer.model_store.set(model_store)
        try:
            module = importlib.import_module(module_name, package=working_dir)
        except ImportError:
            raise ImportServiceError(f'Could not import module "{module_name}"')
        finally:
            # Reset to default local model store
            BentoMLContainer.model_store.set(default_model_store)

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
        if prev_cwd:
            # Reset to previous cwd
            os.chdir(prev_cwd)
        if sys_path_modified and working_dir:
            # Undo changes to sys.path
            sys.path.remove(working_dir)
        raise


@inject
def load_bento(
    bento_tag: str, bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store]
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

    # Use Bento's user project path as working directory when importing the service
    working_dir = bento._fs.getsyspath(BENTO_PROJECT_DIR_NAME)

    # Use Bento's local "{base_dir}/models/" directory as its model store
    model_store = ModelStore(bento._fs.getsyspath("models"))

    svc = import_service(bento.metadata["service"], working_dir, model_store)

    svc.version = bento.metadata["version"]
    svc.tag = bento.metadata.tag
    svc.bento = bento
    return svc


def load(
    svc_import_path_or_bento_tag: str, working_dir: t.Optional[str] = None
) -> "Service":
    """Load a Service instance from source code or a bento in local bento store."""

    try:
        svc = import_service(svc_import_path_or_bento_tag, working_dir)
        logger.info("Imported from source: %s", svc)
    except ImportServiceError as e1:
        try:
            svc = load_bento(svc_import_path_or_bento_tag)
            logger.info("Loaded from Bento: %s", svc)
        except (NotFound, ImportServiceError) as e2:
            raise BentoMLException(
                f"Failed to load bento or import service "
                f"'{svc_import_path_or_bento_tag}'. If you are attempting to "
                f"import bento in local store: `{e1}`, or if you are importing by "
                f"python module path: `{e2}`"
            )
    return svc
