import logging
import shutil
import typing as t
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import attr
import yaml
from simple_di import Provide, inject

from bentoml import __version__ as BENTOML_VERSION

from ...exceptions import InvalidArgument
from ..configuration.containers import BentoMLContainer
from ..types import GenericDictType, ModelTag, PathType
from ..utils import validate_or_create_dir
from . import MODEL_STORE_PREFIX, MODEL_YAML_NAMESPACE, YAML_EXT

logger = logging.getLogger(__name__)

RESERVED_MODEL_FIELD = [
    "name",
    "path",
    "version",
    "module",
    "created_at",
    "labels",
    "context",
]


@attr.s
class StoreCtx(object):
    name = attr.ib(type=str)
    version = attr.ib(type=str)
    path = attr.ib(type=PathType)
    labels = attr.ib(type=t.Dict[str, str], factory=dict, kw_only=True)
    options = attr.ib(type=GenericDictType, factory=dict, kw_only=True)
    metadata = attr.ib(type=GenericDictType, factory=dict, kw_only=True)


@attr.s
class ModelInfo(StoreCtx):
    module = attr.ib(type=str, kw_only=True)
    created_at = attr.ib(type=str, kw_only=True)
    context = attr.ib(type=GenericDictType, factory=dict)
    api_version = attr.ib(init=False, type=str, default="v1")
    bentoml_version = attr.ib(init=False, type=str, default=BENTOML_VERSION)


def dump_model_yaml(
    model_yaml: Path,
    ctx: "StoreCtx",
    *,
    framework_context: dict = None,
    module: str = __name__,
) -> None:
    info = ModelInfo(
        name=ctx.name,
        version=ctx.version,
        labels=ctx.labels,
        options=ctx.options,
        metadata=ctx.metadata,
        path=str(ctx.path),
        created_at=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        context={} if framework_context is None else framework_context,
        module=module,
    )
    with model_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(attr.asdict(info), f)


def load_model_yaml(path: PathType) -> "ModelInfo":
    with Path(path, f"{MODEL_YAML_NAMESPACE}{YAML_EXT}").open(
        "r", encoding="utf-8"
    ) as f:
        info = yaml.safe_load(f)
    return ModelInfo(**info)


class LocalModelStore:
    @inject
    def __init__(self, base_dir: PathType = Provide[BentoMLContainer.bentoml_home]):
        self._BASE_DIR = Path(base_dir, MODEL_STORE_PREFIX)
        validate_or_create_dir(self._BASE_DIR)

    def list_model(self, name: t.Optional[str] = None) -> t.List[str]:
        """
        bentoml models list -> t.List[models name under BENTOML_HOME/models]
        bentoml models list my_nlp_models -> t.List[model_version]
        """
        if not name:
            path = self._BASE_DIR
        else:
            tag = ModelTag.from_str(name)
            path = Path(self._BASE_DIR, tag.name)
        return [_f.name for _f in path.iterdir()]

    def _create_path(self, tag: ModelTag):
        model_path = Path(self._BASE_DIR, tag.name, tag.version)
        validate_or_create_dir(model_path)
        return model_path

    @contextmanager
    def register_model(
        self,
        name: str,
        *,
        module: str = "",
        options: GenericDictType = None,
        metadata: GenericDictType = None,
        labels: GenericDictType = None,
        framework_context: dict = None,
    ) -> "t.Iterator[StoreCtx]":
        """
        with bentoml.models.register(name, options, metadata, labels) as ctx:
            # ctx(model_path, version, metadata)
            model.save(ctx.model_path, metadata=ctx.metadata)
            ctx.metadata["params_a"] = value_a
        """
        model_tag = ModelTag.from_str(name)
        model_path = self._create_path(model_tag)
        model_yaml = Path(model_path, f"{MODEL_YAML_NAMESPACE}{YAML_EXT}")

        ctx = StoreCtx(
            name=model_tag.name,
            path=model_path,
            version=model_tag.version,
            labels=labels,
            metadata=metadata,
            options=options,
        )
        try:
            yield ctx
        except StopIteration:
            # save has failed
            logger.warning(
                f"Failed saving models {str(model_tag)}, deleting {model_path}..."
            )
            shutil.rmtree(model_path)
        finally:
            latest_path = Path(self._BASE_DIR, model_tag.name, "latest")
            dump_model_yaml(
                model_yaml, ctx, framework_context=framework_context, module=module
            )
            if latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(model_path)

    def get_model(self, name: str) -> "ModelInfo":
        """
        bentoml.pytorch.get("my_nlp_model")
        """
        name, version = ModelTag.process_str(name)
        path = Path(self._BASE_DIR, name, version)
        if not path.exists():
            raise FileNotFoundError(
                "Given model name is not found in BentoML model stores. "
                "Make sure to have the correct model name."
            )
        return load_model_yaml(path)

    def delete_model(self, name: str, skip_confirm: bool = False):
        """
        bentoml models delete
        """
        model_name, version = ModelTag.process_str(name)
        basepath = Path(self._BASE_DIR, model_name)
        if ":" not in name:
            basepath.rmdir()
        else:
            path = Path(basepath, version)
            path.rmdir()
            path.unlink(missing_ok=True)

    def push_model(self, name: str):
        ...

    def pull_model(self, name: str):
        ...

    def export_model(self, name: str):
        ...

    def import_model(self, name: str):
        ...

    def __getattribute__(self, item):
        inherited = object.__getattribute__(self, item)
        if "model" in item:

            def wrapped_check_identifier(*args, **kwargs):
                name: str = args[0]  # get_model(name), register_model(name)
                if not name.isidentifier():
                    raise InvalidArgument(
                        f"Invalid model name: '{name}'. A valid identifier "
                        "may only contain letters, numbers, underscores "
                        "and not starting with a number."
                    )
                return inherited(*args, **kwargs)

            return wrapped_check_identifier
        else:
            return inherited


# Global modelstore instance
_model_store = LocalModelStore()

ls = _model_store.list_model
register = _model_store.register_model
delete = _model_store.delete_model
get = _model_store.get_model
