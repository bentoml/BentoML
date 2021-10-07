import importlib
import logging
import os
import sys
import typing as t

import yaml

from bentoml.exceptions import BentoMLException, InvalidArgument, NotFound

if t.TYPE_CHECKING:
    from .service import Service

logger = logging.getLogger(__name__)


class ImportServiceError(BentoMLException):
    pass


def import_service(svc_import_path: str, base_dir: t.Optional[str] = None) -> "Service":
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
        load("fraud_detector.py")
        load("fraud_detector")
    """
    try:
        if base_dir:
            sys.path.insert(0, base_dir)

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

        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            if exc.name != module_str:
                raise exc from None
            raise ImportServiceError(f'Failed importing module "{module_name}".')

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
        raise


def load_bento(bento_tag: str, store) -> "Service":
    """Load a Service instance from a bento found in local bento store:

    Example usage:
        load_bento("FraudDetector:latest")
        load_bento("FraudDetector:20210709_DE14C9")
    """
    # TODO: WIP
    bento_path = store.get(bento_tag)
    bento_config = yaml.load(os.path.join(bento_path, "bento.yml"))
    # check bento_config.bentoml_version, bentoml_version
    # set model_store config
    working_dir = os.path.join(bento_path, bento_config.name)
    svc = import_service(bento_config.svc_import_path, working_dir)
    svc.version = bento_config.version

    return svc


def load(svc_import_path_or_bento_tag: str) -> "Service":
    """Load a Service instance from source code or a bento in local bento store."""
    # TODO: WIP
    if not isinstance(svc_import_path_or_bento_tag, str):
        raise InvalidArgument("bentoml.load argument must be str type")

    try:
        return import_service(svc_import_path_or_bento_tag)
    except ImportServiceError:
        pass

    try:
        load_bento(svc_import_path_or_bento_tag)
    except NotFound:
        pass
