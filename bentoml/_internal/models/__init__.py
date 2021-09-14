import inspect
import typing as t
from enum import Enum, auto
from pathlib import Path

import yaml

from ..types import GenericDictType, PathType

_MT = t.TypeVar("_MT", bound=t.Any)
_T = t.TypeVar("_T")

MODEL_STORE_PREFIX = "models"
SAVE_NAMESPACE = "saved_model"
METADATA_NAMESPACE = "metadata"
MODEL_YAML_NAMESPACE = "model_details"

SAVE_INIT_DOCS = """\
    Save a model instance to BentoML modelstore.

    Examples::
        # train.py
        model = MyPyTorchModel().train()  # type: torch.nn.Module
        ...
        import bentoml.pytorch
        semver = bentoml.pytorch.save("my_nlp_model", model, embedding=128)

    Args:
        name (`str`):
            Name for given model instance.
        model (`Any`):
            Model instance for given frameworks. This can be torch.nn.Module, keras.Model,
            etc.
"""

SAVE_RETURNS_DOCS = """\
    Returns:
        store_name (`str` with a format `name:generated_id`) where `name` is the defined name
        user set for their models, and `generated_id` will be generated UUID by BentoML.
"""

LOAD_INIT_DOCS = """\
    Load a model from BentoML modelstore with given name.

    Examples::
        import bentoml.pytorch           
        # load model
        model = bentoml.pytorch.load("my_nlp_model")

    Args:
        name (`str`):
            Name of a saved model in BentoML modelstore.
"""


class _AutoLowerAttrs(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return f".{name.lower()}"

    def __repr__(self):
        return self._value_


class Extensions(_AutoLowerAttrs):
    H5 = auto()
    HDF5 = auto()
    JSON = auto()
    PKL = auto()
    PTH = auto()
    PT = auto()
    TXT = auto()
    YAML = auto()


def dump_model_metadata(metadata: GenericDictType, path: PathType) -> None:
    metadata_yaml = Path(path, f"{METADATA_NAMESPACE}{Extensions.YAML}")
    with metadata_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f)


def load_model_metadata(path: PathType) -> dict:
    with Path(path, f"{METADATA_NAMESPACE}{Extensions.YAML}").open(
        "r", encoding="utf-8"
    ) as f:
        return yaml.safe_load(f)
