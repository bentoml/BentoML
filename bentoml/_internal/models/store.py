import logging
import typing as t
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import attr
import yaml
from simple_di import Provide, inject

from bentoml import __version__ as BENTOML_VERSION

from ...exceptions import BentoMLException
from ..configuration.containers import BentoMLContainer
from ..types import GenericDictType, ModelTag, PathType
from ..utils import generate_new_version_id, validate_or_create_dir
from . import MODEL_STORE_PREFIX, MODEL_YAML_NAMESPACE

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


# reserved field for model_yaml
# api_version = v1
# created_at=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),

# for YAML file
# api_version = attr.ib(type=str)
# bentoml_version = attr.ib(type=str)

# TODO: will be used with models.get
@attr.s
class StoreCtx:
    name = attr.ib(type=str)
    labels = attr.ib(type=t.Dict[str, str], factory=dict)
    options = attr.ib(type=GenericDictType, factory=dict)
    metadata = attr.ib(type=GenericDictType, factory=dict)


@attr.s
class ModelInfo:
    name = attr.ib(type=str)
    version = attr.ib(type=str)
    module = attr.ib(type=str)
    path = attr.ib(type=PathType)
    created_at = attr.ib(type=str)
    context = attr.ib(type=GenericDictType, factory=dict)
    labels = attr.ib(type=t.Dict[str, str], factory=dict)
    options = attr.ib(type=GenericDictType, factory=dict)
    metadata = attr.ib(type=GenericDictType, factory=dict)


class LocalModelStore:
    @inject
    def __init__(self, base_dir: PathType = Provide[BentoMLContainer.bentoml_home]):
        self._BASE_DIR = Path(base_dir, MODEL_STORE_PREFIX)
        validate_or_create_dir(self._BASE_DIR)

    def _create_model_path(self, tag: ModelTag):
        model_path = Path(self._BASE_DIR, tag.name, tag.version)
        validate_or_create_dir(model_path)
        return model_path

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

    @contextmanager
    def register_model(
        self,
        name: str,
        options: GenericDictType,
        metadata: GenericDictType,
        labels: GenericDictType,
    ) -> "t.Iterator[StoreCtx]":
        """
        with bentoml.models.register(name, options, metadata, labels) as ctx:
            # ctx(model_path, version, metadata)
            model.save(ctx.model_path, metadata=ctx.metadata)
            ctx.metadata["params_a"] = value_a
        """
        model_tag = ModelTag.from_str(name)
        model_path = self._create_model_path(model_tag)
        model_yaml = Path(model_path, MODEL_YAML_NAMESPACE)
        latest_path = Path(self._BASE_DIR, model_tag.name, "latest")

        try:
            yield StoreCtx(
                name=model_tag.name,
                path=model_path,
                version=model_tag.version,
                labels=labels,
                metadata=metadata,
                options=options,
            )
        finally:
            if not model_yaml.is_file():
                pass
            latest_path.unlink()

        latest_path.symlink_to(model_path)
        with model_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(ModelInfo(**options), f)

    def get_model(self, name: str) -> ModelInfo:
        """
        bentoml.pytorch.get("my_nlp_model")
        """
        model_tag = ModelTag.from_str(name)
        path = Path(self._BASE_DIR, model_tag.name, model_tag.version)
        if not path.exists():
            raise FileNotFoundError(
                "Given model name is not found in BentoML model stores. "
                "Make sure to have the correct model name."
            )
        model_yaml = Path(path, MODEL_YAML_NAMESPACE)
        with model_yaml.open("r", encoding="utf-8") as f:
            _info = yaml.safe_load(f)

        return ModelInfo(
            path=path.resolve(),
            module=_info["module"],
            save_options=_info["save_options"],
        )

    def delete_model(self, name: str, skip_confirm: bool = False):
        """
        bentoml models delete
        """
        model_tag = ModelTag.from_str(name)
        basepath = Path(self._BASE_DIR, model_tag.name)
        if ":" not in name:
            basepath.rmdir()
        else:
            path = Path(basepath, model_tag.version)
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


# Global modelstore instance
_model_store = LocalModelStore()

ls = _model_store.list_model
register = _model_store.register_model
delete = _model_store.delete_model
get = _model_store.get_model
