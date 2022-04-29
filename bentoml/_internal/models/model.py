from __future__ import annotations

import typing as t
import logging
from sys import version_info as pyver
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone

import fs
import attr
import yaml
import cattrs
import fs.errors
import fs.mirror
import cloudpickle
from fs.base import FS
from simple_di import inject
from simple_di import Provide

from ..tag import Tag
from ..runner.runnable import BatchDimType
from ..store import Store
from ..store import StoreItem
from ..types import AnyType
from ..types import MetadataDict
from ..utils import label_validator
from ..utils import metadata_validator
from ..runner import Runnable
from ...exceptions import NotFound
from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..types import PathType
    from ..runner import Runner

    ModelSignatureDict: t.TypeAlias = dict[
        str, bool | BatchDimType | AnyType | tuple[AnyType] | None
    ]

logger = logging.getLogger(__name__)

PYTHON_VERSION: str = f"{pyver.major}.{pyver.minor}.{pyver.micro}"
MODEL_YAML_FILENAME = "model.yaml"
CUSTOM_OBJECTS_FILENAME = "custom_objects.pkl"


class ModelOptions:
    @classmethod
    def with_options(cls, **kwargs: t.Any) -> ModelOptions:
        if len(kwargs) != 0:
            for k in kwargs:
                raise ValueError(f"Option {k} is not supported")
        return cls()


@attr.define(repr=False, eq=False)
class Model(StoreItem):
    _tag: Tag
    __fs: FS

    _info: "ModelInfo"
    _custom_objects: t.Optional[t.Dict[str, t.Any]] = None

    _flushed: bool = False

    _runnable: Runnable | None = None

    @staticmethod
    def _export_ext() -> str:
        return "bentomodel"

    @property
    def tag(self) -> Tag:
        return self._tag

    @property
    def _fs(self) -> "FS":
        return self.__fs

    @property
    def info(self) -> "ModelInfo":
        return self._info

    @property
    def custom_objects(self) -> t.Dict[str, t.Any]:
        if self._custom_objects is None:
            if self._fs.isfile(CUSTOM_OBJECTS_FILENAME):
                with self._fs.open(CUSTOM_OBJECTS_FILENAME, "rb") as cofile:
                    self._custom_objects: "t.Optional[t.Dict[str, t.Any]]" = (
                        cloudpickle.load(cofile)
                    )
                    if not isinstance(self._custom_objects, dict):
                        raise ValueError("Invalid custom objects found.")
            else:
                self._custom_objects: "t.Optional[t.Dict[str, t.Any]]" = {}

        return self._custom_objects

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Model) and self._tag == other._tag

    def __hash__(self) -> int:
        return hash(self._tag)

    @staticmethod
    def create(
        name: str,
        *,
        module: str,
        signatures: dict[str, t.Any],
        labels: dict[str, str] | None = None,
        options: ModelOptions = ModelOptions(),
        custom_objects: dict[str, t.Any] | None = None,
        metadata: dict[str, t.Any] | None = None,
        context: ModelContext,
    ) -> "Model":
        """Create a new Model instance in temporary filesystem used for serializing
        model artifacts and save to model store

        Args:
            name: model name in target model store, model version will be automatically
                generated
            module: import path of module used for saving/loading this model, e.g.
                "bentoml.tensorflow"
            labels:  user-defined labels for managing models, e.g. team=nlp, stage=dev
            options: default options for loading this model, defined by runner
                implementation, e.g. xgboost booster_params
            custom_objects: user-defined additional python objects to be saved
                alongside the model, e.g. a tokenizer instance, preprocessor function,
                model configuration json
            metadata: user-defined metadata for storing model training context
                information or model evaluation metrics, e.g. dataset version,
                training parameters, model scores
            context: Environment context managed by BentoML for loading model,
                e.g. {"framework:" "tensorflow", "framework_version": _tf_version}

        Returns:
            object: Model instance created in temporary filesystem
        """
        tag = Tag(name).make_new_version()
        labels = {} if labels is None else labels
        metadata = {} if metadata is None else metadata

        model_fs = fs.open_fs(f"temp://bentoml_model_{name}")

        res = Model(
            tag,
            model_fs,
            ModelInfo(
                tag=tag,
                module=module,
                signatures=signatures,
                labels=labels,
                options=options,
                metadata=metadata,
                context=context,
            ),
            custom_objects,
        )

        return res

    @inject
    def save(
        self, model_store: ModelStore = Provide[BentoMLContainer.model_store]
    ) -> "Model":
        self._save(model_store)

        return self

    def _save(self, model_store: "ModelStore") -> "Model":
        self.flush()

        if not self.validate():
            logger.warning(f"Failed to create Model for {self.tag}, not saving.")
            raise BentoMLException("Failed to save Model because it was invalid")

        with model_store.register(self.tag) as model_path:
            out_fs = fs.open_fs(model_path, create=True, writeable=True)
            fs.mirror.mirror(self._fs, out_fs, copy_if_newer=False)
            self._fs.close()
            self.__fs = out_fs

        logger.info(f"Successfully saved {self}")
        return self

    @classmethod
    def from_fs(cls, item_fs: FS) -> "Model":
        try:
            with item_fs.open(MODEL_YAML_FILENAME, "r") as model_yaml:
                info = ModelInfo.from_yaml_file(model_yaml)
        except fs.errors.ResourceNotFound:
            raise BentoMLException(
                f"Failed to load bento model because it does not contain a '{MODEL_YAML_FILENAME}'"
            )

        res = cls(info.tag, item_fs, info)
        if not res.validate():
            raise BentoMLException(
                f"Failed to load bento model because it contains an invalid '{MODEL_YAML_FILENAME}'"
            )

        res._flushed = True
        return res

    @property
    def path(self) -> str:
        return self.path_of("/")

    def path_of(self, item: str) -> str:
        return self._fs.getsyspath(item)

    def flush(self):
        if not self._flushed:
            self._flush_info()
            self._flush_custom_objects()
        self._flushed = True

    def _flush_info(self):
        with self._fs.open(MODEL_YAML_FILENAME, "w") as model_yaml:
            self.info.dump(model_yaml)

    def _flush_custom_objects(self):
        # pickle custom_objects if it is not None and not empty
        if self.custom_objects:
            with self._fs.open(CUSTOM_OBJECTS_FILENAME, "wb") as cofile:
                cloudpickle.dump(self.custom_objects, cofile)

    @property
    def creation_time(self) -> datetime:
        return self.info.creation_time

    def export(
        self,
        path: str,
        output_format: t.Optional[str] = None,
        *,
        protocol: t.Optional[str] = None,
        user: t.Optional[str] = None,
        passwd: t.Optional[str] = None,
        params: t.Optional[t.Dict[str, str]] = None,
        subpath: t.Optional[str] = None,
    ) -> str:
        self.flush()

        return super().export(
            path,
            output_format,
            protocol=protocol,
            user=user,
            passwd=passwd,
            params=params,
            subpath=subpath,
        )

    def validate(self):
        return self._fs.isfile(MODEL_YAML_FILENAME)

    def __str__(self):
        return f'Model(tag="{self.tag}", path="{self.path}")'

    def to_runner(
        self,
        name: str = "",
        cpu: int | None = None,
        nvidia_gpu: int | None = None,
        custom_resources: dict[str, int | float] | None = None,
        max_batch_size: int | None = None,
        max_latency_ms: int | None = None,
        method_configs: dict[str, dict[str, int]] | None = None,
    ) -> Runner:
        """
        TODO(chaoyu): add docstring

        Args:
            name:
            cpu:
            nvidia_gpu:
            custom_resources:
            max_batch_size:
            max_latency_ms:
            runnable_method_configs:

        Returns:

        """
        return Runner(
            self.to_runnable(),
            name=name if name != "" else self.tag.name,
            models=[self],
            cpu=cpu,
            nvidia_gpu=nvidia_gpu,
            custom_resources=custom_resources,
            max_batch_size=max_batch_size,
            max_latency_ms=max_latency_ms,
            method_configs=method_configs,
        )

    def to_runnable(self) -> t.Type[Runnable]:
        module = __import__(self.info.module)
        if not self._runnable:
            self._runnable = module.get_runnable(self)
        return self._runnable

    def with_options(self, **kwargs: t.Any) -> Model:
        return Model(self._tag, self._fs, self.info.with_options(**kwargs))


class ModelStore(Store[Model]):
    def __init__(self, base_path: "t.Union[PathType, FS]"):
        super().__init__(base_path, Model)


@attr.frozen
class ModelContext:
    framework_name: str
    framework_versions: dict[str, str]
    bentoml_version: str = BENTOML_VERSION
    python_version: str = PYTHON_VERSION

    @staticmethod
    def from_dict(data: dict[str, str | dict[str, str]]) -> ModelContext:
        return cattrs.structure(data, ModelContext)


@attr.frozen
class ModelSignature:
    batchable: bool
    batch_dim: BatchDimType
    input_spec: AnyType | tuple[AnyType, ...] | None
    output_spec: AnyType | None

    @staticmethod
    def from_dict(data: ModelSignatureDict) -> ModelSignature:
        return cattrs.structure(data, ModelSignature)

    @staticmethod
    def convert_dict(data: dict[str, ModelSignatureDict]) -> dict[str, ModelSignature]:
        return {
            k: ModelSignature.from_dict(v) if isinstance(v, dict) else v
            for k, v in data.items()
        }


@attr.define(repr=False, eq=False)
class ModelInfo:
    tag: Tag
    module: str
    labels: t.Dict[str, str] = attr.field(validator=label_validator)
    options: ModelOptions
    metadata: MetadataDict = attr.field(validator=metadata_validator)
    context: ModelContext = attr.field(converter=ModelContext.from_dict)
    signatures: dict[str, ModelSignature] = attr.field(
        converter=ModelSignature.convert_dict
    )
    api_version: str = "v1"
    creation_time: datetime = attr.field(factory=lambda: datetime.now(timezone.utc))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (ModelInfo, FrozenModelInfo)):
            return False

        return (
            self.tag == other.tag
            and self.module == other.module
            and self.signatures == other.signatures
            and self.labels == other.labels
            and self.options == other.options
            and self.metadata == other.metadata
            and self.context == other.context
            and self.signatures == other.signatures
            and self.api_version == other.api_version
            and self.creation_time == other.creation_time
        )

    def __attrs_post_init__(self):
        self.validate()

    def with_options(self, **kwargs: t.Any) -> ModelInfo:
        return ModelInfo(
            tag=self.tag,
            module=self.module,
            signatures=self.signatures,
            labels=self.labels,
            options=self.options.with_options(**kwargs),
            metadata=self.metadata,
            context=self.context,
            api_version=self.api_version,
            creation_time=self.creation_time,
        )

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "name": self.tag.name,
            "version": self.tag.version,
            "creation_time": self.creation_time,
            "api_version": self.api_version,
            "module": self.module,
            "signatures": self.signatures,
            "context": self.context,
            "labels": self.labels,
            "options": self.options,
            "metadata": self.metadata,
        }

    def dump(self, stream: t.IO[t.Any]):
        return yaml.dump(self, stream, sort_keys=False)

    @staticmethod
    def from_yaml_file(stream: t.IO[t.Any]):
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:  # pragma: no cover - simple error handling
            logger.error(exc)
            raise

        if not isinstance(yaml_content, dict):
            raise BentoMLException(f"malformed {MODEL_YAML_FILENAME}")

        yaml_content["tag"] = Tag(
            yaml_content["name"],  # type: ignore
            yaml_content["version"],  # type: ignore
        )
        del yaml_content["name"]
        del yaml_content["version"]

        try:
            model_info = FrozenModelInfo(**yaml_content)  # type: ignore
        except TypeError:  # pragma: no cover - simple error handling
            raise BentoMLException(f"unexpected field in {MODEL_YAML_FILENAME}")
        return model_info

    def validate(self):
        # Validate model.yml file schema, content, bentoml version, etc
        # add tests when implemented
        ...

    def freeze(self) -> "ModelInfo":
        self.__class__ = FrozenModelInfo
        return self


@attr.define(repr=False, eq=False, frozen=True, on_setattr=None)  # type: ignore (pyright doesn't allow for a frozen subclass)
class FrozenModelInfo(ModelInfo):
    pass


def copy_model(
    model_tag: t.Union[Tag, str],
    *,
    src_model_store: ModelStore,
    target_model_store: ModelStore,
):
    """copy a model from src model store to target modelstore, and do nothing if the
    model tag already exist in target model store
    """
    try:
        target_model_store.get(model_tag)  # if model tag already found in target
        return
    except NotFound:
        pass

    model = src_model_store.get(model_tag)
    model.save(target_model_store)


def _ModelInfo_dumper(dumper: yaml.Dumper, info: ModelInfo) -> yaml.Node:
    return dumper.represent_dict(info.to_dict())


yaml.add_representer(ModelInfo, _ModelInfo_dumper)
yaml.add_representer(FrozenModelInfo, _ModelInfo_dumper)
