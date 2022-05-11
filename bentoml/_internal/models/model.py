from __future__ import annotations

import typing as t
import logging
import importlib
from sys import version_info as pyver
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone
from collections import UserDict

import fs
import attr
import yaml
import fs.errors
import fs.mirror
import cloudpickle  # type: ignore (no cloudpickle types)
from fs.base import FS
from cattr.gen import override  # type: ignore (incomplete cattr types)
from cattr.gen import make_dict_unstructure_fn  # type: ignore (incomplete cattr types)
from simple_di import inject
from simple_di import Provide

from ..tag import Tag
from ..store import Store
from ..store import StoreItem
from ..types import MetadataDict
from ..utils import bentoml_cattr
from ..utils import label_validator
from ..utils import metadata_validator
from ..runner import Runner
from ..runner import Runnable
from ...exceptions import NotFound
from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION
from ..runner.runnable import BatchDimType
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..types import AnyType
    from ..types import PathType

    ModelSignatureDict: t.TypeAlias = dict[
        str, bool | BatchDimType | AnyType | tuple[AnyType] | None
    ]


T = t.TypeVar("T")

logger = logging.getLogger(__name__)

PYTHON_VERSION: str = f"{pyver.major}.{pyver.minor}.{pyver.micro}"
MODEL_YAML_FILENAME = "model.yaml"
CUSTOM_OBJECTS_FILENAME = "custom_objects.pkl"


if TYPE_CHECKING:
    ModelOptionsSuper = UserDict[str, t.Any]
else:
    ModelOptionsSuper = UserDict


class ModelOptions(ModelOptionsSuper):
    @classmethod
    def with_options(cls, **kwargs: t.Any) -> ModelOptions:
        if len(kwargs) != 0:
            for k in kwargs:
                raise ValueError(f"Option {k} is not supported")
        return cls()

    @staticmethod
    def to_dict(options: ModelOptions) -> dict[str, t.Any]:
        return dict(options)


bentoml_cattr.register_structure_hook_func(
    lambda cls: issubclass(cls, ModelOptions), lambda d, cls: cls(**d)
)
bentoml_cattr.register_unstructure_hook(ModelOptions, ModelOptions.to_dict)


@attr.define(repr=False, eq=False, init=False)
class Model(StoreItem):
    _tag: Tag
    __fs: FS

    _info: ModelInfo
    _custom_objects: dict[str, t.Any] | None = None

    _runnable: Runnable | None = None

    def __init__(
        self,
        tag: Tag,
        fs: FS,
        info: ModelInfo,
        custom_objects: dict[str, t.Any] | None = None,
        runnable: Runnable | None = None,
        *,
        _internal: bool = False,
    ):
        if not _internal:
            raise BentoMLException(
                "Model cannot be instantiated directly directly; use bentoml.<framework>.save or bentoml.models.get instead"
            )

        self.__attrs_init__(tag, fs, info, custom_objects)

    # for type checking purposes
    def __attrs_init__(
        self,
        tag: Tag,
        fs: FS,
        info: ModelInfo,
        custom_objects: dict[str, t.Any] | None = None,
        flushed: bool = False,
        runnable: Runnable | None = None,
    ):
        ...

    @staticmethod
    def _export_ext() -> str:
        return "bentomodel"

    @property
    def tag(self) -> Tag:
        return self._tag

    @property
    def _fs(self) -> FS:
        return self.__fs

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def custom_objects(self) -> t.Dict[str, t.Any]:
        if self._custom_objects is None:
            if self._fs.isfile(CUSTOM_OBJECTS_FILENAME):
                with self._fs.open(CUSTOM_OBJECTS_FILENAME, "rb") as cofile:
                    self._custom_objects: t.Optional[
                        t.Dict[str, t.Any]
                    ] = cloudpickle.load(cofile)
                    if not isinstance(self._custom_objects, dict):
                        raise ValueError("Invalid custom objects found.")
            else:
                self._custom_objects: dict[str, t.Any] | None = {}

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
        options: ModelOptions | None = None,
        custom_objects: dict[str, t.Any] | None = None,
        metadata: dict[str, t.Any] | None = None,
        context: ModelContext,
    ) -> Model:
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
        options = ModelOptions() if options is None else options

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
            custom_objects=custom_objects,
            _internal=True,
        )

        return res

    @inject
    def save(
        self, model_store: ModelStore = Provide[BentoMLContainer.model_store]
    ) -> Model:
        self._save(model_store)

        return self

    def _save(self, model_store: ModelStore) -> Model:
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
    def from_fs(cls: t.Type[Model], item_fs: FS) -> Model:
        try:
            with item_fs.open(MODEL_YAML_FILENAME, "r") as model_yaml:
                info = ModelInfo.from_yaml_file(model_yaml)
        except fs.errors.ResourceNotFound:
            raise BentoMLException(
                f"Failed to load bento model because it does not contain a '{MODEL_YAML_FILENAME}'"
            )

        res = Model(tag=info.tag, fs=item_fs, info=info, _internal=True)
        if not res.validate():
            raise BentoMLException(
                f"Failed to load bento model because it contains an invalid '{MODEL_YAML_FILENAME}'"
            )

        return res

    @property
    def path(self) -> str:
        return self.path_of("/")

    def path_of(self, item: str) -> str:
        return self._fs.getsyspath(item)

    def flush(self):
        self._write_info()
        self._write_custom_objects()

    def _write_info(self):
        with self._fs.open(MODEL_YAML_FILENAME, "w") as model_yaml:
            self.info.dump(model_yaml)

    def _write_custom_objects(self):
        # pickle custom_objects if it is not None and not empty
        if self.custom_objects:
            with self._fs.open(CUSTOM_OBJECTS_FILENAME, "wb") as cofile:
                cloudpickle.dump(self.custom_objects, cofile)  # type: ignore (incomplete cloudpickle types)

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
        custom_resources: dict[str, float] | None = None,
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
        module = importlib.import_module(self.info.module)
        if not self._runnable:
            self._runnable = module.get_runnable(self)
        return self._runnable

    def with_options(self, **kwargs: t.Any) -> Model:
        res = Model(
            self._tag,
            self._fs,
            self.info.with_options(**kwargs),
            self._custom_objects,
            _internal=True,
        )
        res.__attrs_init__(
            self._tag,
            self._fs,
            self.info.with_options(**kwargs),
            self._custom_objects,
        )
        return res


class ModelStore(Store[Model]):
    def __init__(self, base_path: "t.Union[PathType, FS]"):
        super().__init__(base_path, Model)


@attr.frozen
class ModelContext:
    framework_name: str
    framework_versions: t.Dict[str, str]
    bentoml_version: str = attr.field(default=BENTOML_VERSION)
    python_version: str = attr.field(default=PYTHON_VERSION)

    @staticmethod
    def from_dict(data: dict[str, str | dict[str, str]] | ModelContext) -> ModelContext:
        if isinstance(data, ModelContext):
            return data
        return bentoml_cattr.structure(data, ModelContext)


# Remove after attrs support ForwardRef natively
attr.resolve_types(ModelContext, globals(), locals())


@attr.frozen
class ModelSignature:
    """
    A model signature represents a method on a model object that can be called.

    Note that anywhere a ``ModelSignature`` is used, a ``dict`` with keys corresponding to the fields
    can be used instead. For example, instead of ``{"predict": ModelSignature(batchable=True)}``, one
    can pass ``{"predict": {"batchable": True}}``.

    Fields:
        batchable:
            Whether the method should support batching. Note that batching must also
            be explicitly enabled at runtime in the ``Model.to_runner`` or ``bentoml.runner`` call.
        batch_dim:
            The dimension(s) that contain multiple data. For example, if when passing two input arrays
            ``[1, 2]`` and ``[5, 6]``, if the input array is ``[[1, 2], [3, 4]]``, then the batch dimension
            would be ``0``. If the input array is ``[[1, 3], [2, 4]]``, then the batch dimension would be ``1``.

            Expert API:

            The batch dimension can also be a tuple of (input batch dimension, output batch dimension). In this
            case, the input batch dimension can be an array which contains the batch dimension for each argument.
            For example, if the predict method takes three arguments ``predict(x, y, z)``, the batch dimension could be
            ``([0, 1, 2], 0)``, meaning ``x`` would be batched along the zeroth dimension, ``y`` along the first,
            ``z`` along the second, and the output along the zeroth axis.

            Partial argument arrays are not supported; if the input takes n arguments, the batch dimension array must
            have length n.
        input_spec: Reserved for future use.
        output_spec: Reserved for future use.
    """

    batchable: bool = False
    batch_dim: BatchDimType = 0
    # TODO: define input/output spec struct
    input_spec: t.Any = None
    output_spec: t.Any = None

    @staticmethod
    def from_dict(data: t.Dict[str, t.Any]) -> ModelSignature:
        return bentoml_cattr.structure(data, ModelSignature)

    @staticmethod
    def convert_signatures_dict(
        data: dict[str, ModelSignatureDict | ModelSignature]
    ) -> dict[str, ModelSignature]:
        return {
            k: ModelSignature.from_dict(v) if isinstance(v, dict) else v
            for k, v in data.items()
        }


# Remove after attrs support ForwardRef natively
attr.resolve_types(ModelSignature, globals(), locals())


def model_signature_encoder(model_signature: ModelSignature) -> dict[str, t.Any]:
    encoded: dict[str, t.Any] = {
        "batchable": model_signature.batchable,
    }
    # ignore batch_dim if batchable is False
    if model_signature.batchable:
        encoded["batch_dim"] = model_signature.batch_dim
    if model_signature.input_spec is not None:
        encoded["input_spec"] = model_signature.input_spec
    if model_signature.output_spec is not None:
        encoded["output_spec"] = model_signature.output_spec
    return encoded


bentoml_cattr.register_unstructure_hook(ModelSignature, model_signature_encoder)


@attr.define(repr=False, eq=False)
class ModelInfo:
    tag: Tag
    name: str = attr.field(init=False)  # converted from tag in __attrs_post_init__
    version: str = attr.field(init=False)  # converted from tag in __attrs_post_init__
    module: str
    labels: t.Dict[str, str] = attr.field(validator=label_validator)
    options: ModelOptions
    metadata: MetadataDict = attr.field(validator=metadata_validator, converter=dict)
    context: ModelContext = attr.field()
    signatures: t.Dict[str, ModelSignature] = attr.field(
        converter=ModelSignature.convert_signatures_dict
    )
    api_version: str = attr.field(default="v1")
    creation_time: datetime = attr.field(factory=lambda: datetime.now(timezone.utc))

    def __attrs_post_init__(self):
        object.__setattr__(self, "name", self.tag.name)
        object.__setattr__(self, "version", self.tag.version)

        self.validate()

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
        return bentoml_cattr.unstructure(self)  # type: ignore (incomplete cattr types)

    def dump(self, stream: t.IO[t.Any]):
        return yaml.dump(self.to_dict(), stream, sort_keys=False)

    @staticmethod
    def from_yaml_file(stream: t.IO[t.Any]):
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:  # pragma: no cover - simple error handling
            logger.error(exc)
            raise

        if not isinstance(yaml_content, dict):
            raise BentoMLException(f"malformed {MODEL_YAML_FILENAME}")

        yaml_content["tag"] = str(
            Tag(
                yaml_content["name"],  # type: ignore
                yaml_content["version"],  # type: ignore
            )
        )
        del yaml_content["name"]
        del yaml_content["version"]

        # For backwards compatibility for bentos created prior to version 1.0.0rc1
        if "bentoml_version" in yaml_content:
            del yaml_content["bentoml_version"]
        if "signatures" not in yaml_content:
            yaml_content["signatures"] = {}

        try:
            model_info = bentoml_cattr.structure(yaml_content, FrozenModelInfo)
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


# Remove after attrs support ForwardRef natively
attr.resolve_types(ModelInfo, globals(), locals())
attr.resolve_types(FrozenModelInfo, globals(), locals())

bentoml_cattr.register_unstructure_hook_func(
    lambda cls: issubclass(cls, ModelInfo),  # for both ModelInfo and FrozenModelInfo
    # Ignore tag, tag is saved via the name and version field
    make_dict_unstructure_fn(ModelInfo, bentoml_cattr, tag=override(omit=True)),  # type: ignore (incomplete types)
)


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


yaml.add_representer(ModelInfo, _ModelInfo_dumper)  # type: ignore (incomplete yaml types)
yaml.add_representer(FrozenModelInfo, _ModelInfo_dumper)  # type: ignore (incomplete yaml types)
