from __future__ import annotations

import typing as t
import logging

import attr

from .... import models
from ...types import PathType
from ...utils import LazyLoader
from ...runner import Runner
from ....exceptions import NotFound
from ...models.model import Model
from ...models.model import ModelContext
from ...models.model import PartialKwargsModelOptions as ModelOptions
from ...runner.runnable import Runnable
from ...runner.runner_handle import RunnerHandle

if t.TYPE_CHECKING:
    from types import ModuleType

    from google.protobuf import text_format as pbtxt

    from ...tag import Tag
    from ...models.model import ModelSignatureDict
    from . import model_config_pb2 as pb_model_config

    TritonBackendType = t.Literal[
        "pytorch",
        "tensorflow1",  # Currently BentoML doesn't provide tensorflow 1 support
        "tensorflow",
        "onnxruntime",
        "tensorrt",
        "python",
        "openvino",
        "paddle",
        "fil",
    ]
else:
    pbtxt = LazyLoader(
        "pbtxt",
        globals(),
        "google.protobuf.text_format",
        exc_msg="'protobuf' is required to use triton with BentoML. Install with 'pip install bentoml[triton]'.",
    )

    pb_model_config = LazyLoader(
        "pb_model_config",
        globals(),
        "bentoml._internal.integrations.triton.model_config_pb2",
    )

MODULE_NAME = "bentoml.triton"
MODEL_FILENAME = "model"
API_VERSION = "v1"

logger = logging.getLogger(__name__)


def get(tag_like: str | Tag) -> Model:
    model = models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def _pbtxt_converter(pbtxt: PathType | dict[str, t.Any]) -> dict[str, t.Any]:
    if not isinstance(pbtxt, dict):
        pass
    return pbtxt


@attr.define
class TritonModelOptions(ModelOptions):
    config_pbtxt: PathType | t.Dict[str, t.Any] = attr.field(
        factory=dict, converter=_pbtxt_converter
    )


def save_model(
    name: str,
    model: t.Any,
    backend: TritonBackendType,
    config_pbtxt: PathType | dict[str, t.Any] | None = None,
    model_labels: PathType | dict[str, t.Any] | None = None,
    *,
    signatures: dict[str, ModelSignatureDict] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: list[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
    **kwargs: t.Any,
):
    ...


def create_runners_from_repository(path: PathType) -> TritonRunnerRepository:
    ...


@attr.define
class TritonRunnerRepository:
    ...


def create_runner(**kwargs: t.Any) -> TritonRunner:
    ...


class TritonRunnable(Runnable):
    ...


class TritonRunner(Runner):
    ...


class TritonRunnerHandle(RunnerHandle):
    ...
