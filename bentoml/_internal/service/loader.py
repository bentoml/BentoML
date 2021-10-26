import importlib
import logging
import os
import sys
import typing as t

import yaml
from simple_di import Provide, inject

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models.store import ModelStore
from bentoml.exceptions import BentoMLException, InvalidArgument, NotFound

if t.TYPE_CHECKING:
    from .service import Service

logger = logging.getLogger(__name__)


class ImportServiceError(BentoMLException):
    pass


@inject
def import_service(
    svc_import_path: str,
    base_dir: t.Optional[str] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "Service":
    """Import a Service instance from source code, by providing the svc_import_path
    which represents the module where the Service instance is created and optionally
    what attribute can be used to access this Service instance in that module

    Example usage:
        import_service("fraud_detector:svc")
        import_service("foo.bar.fraud_detector:svc")
        import_service("fraud_detector:foo.bar.svc")
        import_service("fraud_detector.py:svc")

        # When there's only one Service instance in the target module, the attributes
        # part in the svc_import_path can be omitted
        import_service("fraud_detector.py")
        import_service("fraud_detector")
    """
    try:
        if base_dir:
            sys.path.insert(0, base_dir)
            # Set cwd(current working directory) to the Bento's project directory, which
            # allows user code to read files using relative path
            prev_cwd = os.getcwd()
            os.chdir(base_dir)

        module_str, _, attrs_str = svc_import_path.partition(":")
        if not module_str:
            raise ImportServiceError(
                f'Invalid import target "{svc_import_path}", must format as '
                '"<module>:<attribute>" or "<module>'
            )

        module_name, ext = os.path.splitext(module_str)
        if ext and ext != ".py":
            raise ImportServiceError(
                f'Invalid module extension "{ext}" found in target "{svc_import_path}",'
                ' the only extension acceptable here is ".py"'
            )

        default_model_store = BentoMLContainer.model_store
        try:
            # Import the service using the Bento's own model store
            BentoMLContainer.model_store.set(model_store)

            module = importlib.import_module(module_name)
        except ImportError as exc:
            if exc.name != module_str:
                raise exc from None
            raise ImportServiceError(f'Failed importing module "{module_name}".')
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
            instances = [v for v in module.__dict__.values() if isinstance(v, Service)]

            if len(instances) == 1:
                instance = instances[0]
            else:
                raise ImportServiceError(
                    f'Multiple Service instances found in module "{module_name}", use'
                    '"<module>:<svc_variable_name>" to specify the service instance or'
                    "define only service instance per python module/file"
                )

        if base_dir:
            instance._working_dir = base_dir
        return instance
    except ImportServiceError:
        if base_dir:
            sys.path.remove(base_dir)
            # Reset to previous cwd
            os.chdir(prev_cwd)
        raise


@inject
def load_bento(
    bento_tag: str, bento_store=Provide[BentoMLContainer.bento_store]
) -> "Service":
    """Load a Service instance from a bento found in local bento store:

    Example usage:
        load_bento("FraudDetector:latest")
        load_bento("FraudDetector:20210709_DE14C9")
    """
    bento_info = bento_store.get(bento_tag)
    bento_config = yaml.load(os.path.join(bento_info.path, "bento.yml"))
    # TODO: verify bento_config.bentoml_version, bentoml_version

    # Use Bento's user project path as working directory when importing the service
    working_dir = os.path.join(bento_info.path, bento_config.name)
    # Use Bento's local "{base_dir}/models/" directory as its model store
    model_store = ModelStore(os.path.join(bento_info.path, "models"))

    svc = import_service(bento_config.svc_import_path, working_dir, model_store)
    svc.version = bento_config.version
    return svc


def load(svc_import_path_or_bento_tag: str, working_dir: t.Optional[str]) -> "Service":
    """Load a Service instance from source code or a bento in local bento store."""
    if not isinstance(svc_import_path_or_bento_tag, str):
        raise InvalidArgument("bentoml.load argument must be str type")

    if working_dir:
        return import_service(svc_import_path_or_bento_tag, working_dir)

    try:
        load_bento(svc_import_path_or_bento_tag)
    except (NotFound, ImportServiceError) as e1:
        try:
            return import_service(svc_import_path_or_bento_tag)
        except ImportServiceError as e2:
            raise BentoMLException(
                f"Failed to load bento or import service "
                f"'{svc_import_path_or_bento_tag}', make sure Bento exist in local "
                f'store with the `bentoml.get("my_svc:20211023_CB341A")` or make '
                f'sure the import path is correct, e.g. "./bento.py:svc". {e1}, {e2}'
            )
