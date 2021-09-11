import logging
import typing as t
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import attr
import yaml
from simple_di import Provide, inject

from bentoml import __version__ as BENTOML_VERSION

from ..configuration.containers import BentoMLContainer
from ..types import GenericDictType, PathType
from ..utils import generate_new_version_id, validate_or_create_dir

LOCAL_MODELSTORE_NAMESPACE = "models"
BENTOML_MODEL_YAML = "bentoml_model.yaml"

logger = logging.getLogger(__name__)


def _process_name(model_name: str, sep: str = ":") -> t.Tuple[str, str]:
    # name can be my_model, my_model:latest, my_model:20210907084629_36AE55
    if sep not in model_name:
        return model_name, "latest"
    else:
        name, version = model_name.split(sep)
        if not version:
            # in case users define name in format `my_nlp_model:`
            logger.warning(
                f"{model_name} contains leading `:`. Make sure to "
                "have correct name format (my_nlp_model, my_nlp_model:latest"
                ", and my_nlp_model:20210907_36AE55). Assuming defaults to"
                " `latest`..."
            )
            version = "latest"
        return name, version

# reserved field for model_yaml

def _gen_model_yaml(
    *,
    module=None,
    save_options=None,
    metadata: GenericDictType = None,
    stacks: str = None,
    **context_kwargs,
) -> dict:
    return dict(
        api_version="v1",
        bentoml_version=BENTOML_VERSION,
        created_at=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        module=module,
        save_options=save_options,
        metadata=metadata,
        context=dict(source=stacks, **context_kwargs),
    )

# for YAML file
# api_version = attr.ib(type=str)
# bentoml_version = attr.ib(type=str)

# TODO: will be used with models.get
@attr.s
class ModelDetails:
    name = attr.ib(type=str)
    version = attr.ib(type=str)
    module = attr.ib(type=str)
    created_at = attr.ib(type=str)
    labels = attr.ib(type=t.Dict[str, str], factory=dict)
    context = attr.ib(type=GenericDictType, factory=dict)
    save_options = attr.ib(type=GenericDictType, factory=dict)
    metadata = attr.ib(type=GenericDictType, factory=dict)


@attr.s
class ModelInfo:
    path = attr.ib(type=PathType)
    module = attr.ib(type=str)
    save_options = attr.ib(type=GenericDictType, factory=dict)


@attr.s
class StoreCtx:
    path = attr.ib(type=PathType)
    version = attr.ib(type=str)
    metadata = attr.ib(type=GenericDictType, factory=dict)


class LocalModelStore:
    @inject
    def __init__(self, base_dir: PathType = Provide[BentoMLContainer.bentoml_home]):
        self._BASE_DIR = Path(base_dir, LOCAL_MODELSTORE_NAMESPACE)
        validate_or_create_dir(self._BASE_DIR)

    def list_model(self, name: t.Optional[str] = None) -> t.List[str]:
        """
        bentoml models list -> t.List[models name under BENTOML_HOME/models]
        bentoml models list my_nlp_models -> t.List[model_version]
        """
        if not name:
            path = self._BASE_DIR
        else:
            name, version = _process_name(name)
            path = Path(self._BASE_DIR, name)
        return [_f.name for _f in path.iterdir()]

    @contextmanager
    def add_model(self, name: str, **yaml_kwargs: str) -> t.Iterator[StoreCtx]: # register_model
        """
        with bentoml.models.add(name, module, options) as ctx:
            # ctx(path, version, metadata)
            model.save(ctx.path)
            ctx.metadata["params_a"] = value_a
        """
        version = generate_new_version_id()
        path = Path(self._BASE_DIR, name, version)
        validate_or_create_dir(path)
        latest_path = Path(self._BASE_DIR, name, "latest")

        try:
            latest_path.unlink()
        except FileNotFoundError:
            pass

        latest_path.symlink_to(path)
        # save bentoml_model.yml
        model_yaml = Path(path, BENTOML_MODEL_YAML)
        with model_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(_gen_model_yaml(**yaml_kwargs), f)

        yield StoreCtx(path=path, version=version)

    def get_model(self, name: str) -> ModelInfo:
        """
        bentoml.pytorch.load("my_nlp_model")
        """
        name, version = _process_name(name)
        path = Path(self._BASE_DIR, name, version)
        if not path.exists():
            raise FileNotFoundError(
                "Given model name is not found in BentoML model stores. "
                "Make sure to have the correct model name."
            )
        model_yaml = Path(path, BENTOML_MODEL_YAML)
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
        basepath = Path(self._BASE_DIR, name)
        if ":" not in name:
            basepath.rmdir()
        else:
            name, version = _process_name(name)
            if version == "latest":
                # covers cases where users mistype
                # leading ":"
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


# Global modelstore instance
modelstore = LocalModelStore()

ls = modelstore.list_model
add = modelstore.add_model # register model
delete = modelstore.delete_model
get = modelstore.get_model