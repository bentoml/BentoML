from __future__ import annotations

import io
import os
import typing as t
import logging
import importlib
from sys import version_info as pyver
from types import ModuleType
from typing import overload
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone

import fs
import attr
import yaml
import fs.errors
import fs.mirror
import cloudpickle  # type: ignore (no cloudpickle types)
from fs.base import FS
from cattr.gen import override
from cattr.gen import make_dict_structure_fn
from cattr.gen import make_dict_unstructure_fn
from simple_di import inject
from simple_di import Provide

from ..tag import Tag
from ..store import Store
from ..store import StoreItem
from ..types import MetadataDict
from ..utils import bentoml_cattr
from ..utils import label_validator
from ..utils import metadata_validator
from ..utils import normalize_labels_value
from ...exceptions import NotFound
from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION
from ..configuration.containers import BentoMLContainer
from ..types import ModelSignatureDict

if t.TYPE_CHECKING:
    from ..types import PathType
    from ..runner import Runner
    from ..runner import Runnable
    from ..runner.strategy import Strategy


T = t.TypeVar("T")

logger = logging.getLogger(__name__)

PYTHON_VERSION: str = f"{pyver.major}.{pyver.minor}.{pyver.micro}"
MODEL_YAML_FILENAME = "model.yaml"
CUSTOM_OBJECTS_FILENAME = "custom_objects.pkl"


@attr.define
class ModelOptions:
    def with_options(self, **kwargs: t.Any) -> ModelOptions:
        return attr.evolve(self, **kwargs)

    def to_dict(self: ModelOptions) -> dict[str, t.Any]:
        return attr.asdict(self)


@attr.define
class PartialKwargsModelOptions(ModelOptions):
    partial_kwargs: t.Dict[str, t.Any] = attr.field(factory=dict)


@attr.define(repr=False, eq=False, init=False)
class Model(StoreItem):
    _tag: Tag
    __fs: FS

    _info: ModelInfo
    _custom_objects: dict[str, t.Any] | None = None

    _runnable: t.Type[Runnable] | None = attr.field(init=False, default=None)

    _model: t.Any = None

    def __init__(
        self,
        tag: Tag,
        model_fs: FS,
        info: ModelInfo,
        custom_objects: dict[str, t.Any] | None = None,
        *,
        _internal: bool = False,
    ):
        if not _internal:
            raise BentoMLException(
                "Model cannot be instantiated directly directly; use bentoml.<framework>.save or bentoml.models.get instead"
            )

        self.__attrs_init__(tag, model_fs, info, custom_objects)  # type: ignore (no types for attrs init)

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
                    self._custom_objects: dict[str, t.Any] | None = cloudpickle.load(
                        cofile
                    )
                    if not isinstance(self._custom_objects, dict):
                        raise ValueError("Invalid custom objects found.")
            else:
                self._custom_objects: dict[str, t.Any] | None = {}

        return self._custom_objects

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Model) and self._tag == other._tag

    def __hash__(self) -> int:
        return hash(self._tag)

    @classmethod
    def create(
        cls,
        name: Tag | str,
        *,
        module: str,
        api_version: str,
        signatures: ModelSignaturesType,
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
        tag = Tag.from_taglike(name)
        if tag.version is None:
            tag = tag.make_new_version()
        labels = {} if labels is None else labels
        metadata = {} if metadata is None else metadata
        options = ModelOptions() if options is None else options

        model_fs = fs.open_fs(f"temp://bentoml_model_{tag.name}")

        return cls(
            tag,
            model_fs,
            ModelInfo(
                tag=tag,
                module=module,
                api_version=api_version,
                signatures=signatures,
                labels=labels,
                options=options,
                metadata=metadata,
                context=context,
            ),
            custom_objects=custom_objects,
            _internal=True,
        )

    @inject
    def save(
        self,
        model_store: ModelStore = Provide[BentoMLContainer.model_store],
    ) -> Model:
        try:
            self.validate()
        except BentoMLException as e:
            raise BentoMLException(f"Failed to save {self!s}: {e}") from None

        with model_store.register(self.tag) as model_path:
            out_fs = fs.open_fs(model_path, create=True, writeable=True)
            fs.mirror.mirror(self._fs, out_fs, copy_if_newer=False)
            self._fs.close()
            self.__fs = out_fs

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

        res = Model(tag=info.tag, model_fs=item_fs, info=info, _internal=True)
        try:
            res.validate()
        except BentoMLException as e:
            raise BentoMLException(f"Failed to load {res!s}: {e}") from None

        return res

    @property
    def path(self) -> str:
        return self.path_of("/")

    def path_of(self, item: str) -> str:
        return self._fs.getsyspath(item)

    @classmethod
    def enter_cloudpickle_context(
        cls,
        external_modules: list[ModuleType],
        imported_modules: list[ModuleType],
    ) -> list[ModuleType]:
        """
        Enter a context for cloudpickle to pickle custom objects defined in external modules.

        Args:
            external_modules: list of external modules to pickle
            imported_modules: list to added modules, needs to be surely unregistered after pickling

        Returns:
            list[ModuleType]: list of module names that were imported in the context

        Raise:
            ValueError: if any of the external modules is not importable
        """
        if not external_modules:
            return []

        registed_before: set[str] = cloudpickle.list_registry_pickle_by_value()
        for mod in external_modules:
            if mod.__name__ in registed_before:
                continue
            cloudpickle.register_pickle_by_value(mod)
            imported_modules.append(mod)

        return imported_modules

    @classmethod
    def exit_cloudpickle_context(cls, imported_modules: list[ModuleType]) -> None:
        """
        Exit the context for cloudpickle, unregister imported external modules.

        Needs to be called after self.flush
        """
        if not imported_modules:
            return

        for mod in imported_modules:
            cloudpickle.unregister_pickle_by_value(mod)

    def flush(self):
        self._write_info()
        self._write_custom_objects()

    def _write_info(self):
        with self._fs.open(MODEL_YAML_FILENAME, "w", encoding="utf-8") as model_yaml:
            self.info.dump(t.cast(io.StringIO, model_yaml))

    def _write_custom_objects(self):
        # pickle custom_objects if it is not None and not empty
        if self.custom_objects:
            with self._fs.open(CUSTOM_OBJECTS_FILENAME, "wb") as cofile:
                cloudpickle.dump(self.custom_objects, cofile)  # type: ignore (incomplete cloudpickle types)

    @property
    def creation_time(self) -> datetime:
        return self.info.creation_time

    def validate(self):
        if not self._fs.isfile(MODEL_YAML_FILENAME):
            raise BentoMLException(
                f"{self!s} does not contain a {MODEL_YAML_FILENAME}."
            )

    def __str__(self):
        return f'Model(tag="{self.tag}")'

    def __repr__(self):
        return f'Model(tag="{self.tag}", path="{self.path}")'

    def to_runner(
        self,
        name: str = "",
        max_batch_size: int | None = None,
        max_latency_ms: int | None = None,
        method_configs: dict[str, dict[str, int]] | None = None,
        embedded: bool = False,
        scheduling_strategy: type[Strategy] | None = None,
    ) -> Runner:
        """
        TODO(chaoyu): add docstring

        Args:
            name:
            max_batch_size:
            max_latency_ms:
            runnable_method_configs:

        Returns:

        """
        from ..runner import Runner
        from ..runner.strategy import DefaultStrategy

        if scheduling_strategy is None:
            scheduling_strategy = DefaultStrategy

        # TODO: @larme @yetone run this branch only yatai version is incompatible with embedded runner
        yatai_version = os.environ.get("YATAI_T_VERSION")
        if embedded and yatai_version:
            logger.warning(
                f"Yatai of version {yatai_version} is incompatible with embedded runner, set `embedded=False` for runner {name}"
            )
            embedded = False

        return Runner(
            self.to_runnable(),
            name=name if name != "" else self.tag.name,
            models=[self],
            max_batch_size=max_batch_size,
            max_latency_ms=max_latency_ms,
            method_configs=method_configs,
            embedded=embedded,
            scheduling_strategy=scheduling_strategy,
        )

    def to_runnable(self) -> t.Type[Runnable]:
        if self._runnable is None:
            self._runnable = self.info.imported_module.get_runnable(self)
        return self._runnable

    def load_model(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """
        Load the model into memory from the model store directory.
        This is a shortcut to the ``load_model`` function defined in the framework module
        used for saving the target model.

        For example, if the ``BentoModel`` is saved with
        ``bentoml.tensorflow.save_model``, this method will pass it to the
        ``bentoml.tensorflow.load_model`` method, along with any additional arguments.
        """
        if self._model is None:
            self._model = self.info.imported_module.load_model(self, *args, **kwargs)
        return self._model

    def with_options(self, **kwargs: t.Any) -> Model:
        res = Model(
            self._tag,
            self._fs,
            self.info.with_options(**kwargs),
            self._custom_objects,
            _internal=True,
        )
        return res


class ModelStore(Store[Model]):
    def __init__(self, base_path: "t.Union[PathType, FS]"):
        super().__init__(base_path, Model)


@attr.frozen
class ModelContext:
    framework_name: str
    framework_versions: t.Dict[str, str]

    # using factory explicitly instead of default because omit_if_default is enabled in ModelInfo
    bentoml_version: str = attr.field(factory=lambda: BENTOML_VERSION)
    python_version: str = attr.field(factory=lambda: PYTHON_VERSION)

    @staticmethod
    def from_dict(data: dict[str, str | dict[str, str]] | ModelContext) -> ModelContext:
        if isinstance(data, ModelContext):
            return data
        return bentoml_cattr.structure(data, ModelContext)

    def to_dict(self: ModelContext) -> dict[str, str | dict[str, str]]:
        return bentoml_cattr.unstructure(self)


@attr.frozen
class ModelSignature:
    """
    A model signature represents a method on a model object that can be called.

    This information is used when creating BentoML runners for this model.

    Note that anywhere a ``ModelSignature`` is used, a ``dict`` with keys corresponding to the
    fields can be used instead. For example, instead of ``{"predict":
    ModelSignature(batchable=True)}``, one can pass ``{"predict": {"batchable": True}}``.

    Fields:
        batchable:
            Whether multiple API calls to this predict method should be batched by the BentoML
            runner.
        batch_dim:
            The dimension(s) that contain multiple data when passing to this prediction method.

            For example, if you have two inputs you want to run prediction on, ``[1, 2]`` and ``[3,
            4]``, if the array you would pass to the predict method would be ``[[1, 2], [3, 4]]``,
            then the batch dimension would be ``0``. If the array you would pass to the predict
            method would be ``[[1, 3], [2, 4]]``, then the batch dimension would be ``1``.

            If there are multiple arguments to the predict method and there is only one batch
            dimension supplied, all arguments will use that batch dimension.

            Example: .. code-block:: python
                # Save two models with `predict` method that supports taking input batches on the
                dimension 0 and the other on dimension 1: bentoml.pytorch.save_model("demo0",
                model_0, signatures={"predict": {"batchable": True, "batch_dim": 0}})
                bentoml.pytorch.save_model("demo1", model_1, signatures={"predict": {"batchable":
                True, "batch_dim": 1}})
                # if the following calls are batched, the input to the actual predict method on the
                # model.predict method would be [[1, 2], [3, 4], [5, 6]] runner0 =
                bentoml.pytorch.get("demo0:latest").to_runner() runner0.init_local()
                runner0.predict.run(np.array([[1, 2], [3, 4]])) runner0.predict.run(np.array([[5,
                6]]))
                # if the following calls are batched, the input to the actual predict method on the
                # model.predict would be [[1, 2, 5], [3, 4, 6]] runner1 =
                bentoml.pytorch.get("demo1:latest").to_runner() runner1.init_local()
                runner1.predict.run(np.array([[1, 2], [3, 4]])) runner1.predict.run(np.array([[5],
                [6]]))

            Expert API:

            The batch dimension can also be a tuple of (input batch dimension, output batch
            dimension). For example, if the predict method should have its input batched along the
            first axis and its output batched along the zeroth axis, ``batch_dim`` can be set to
            ``(1, 0)``.

        input_spec: Reserved for future use.

        output_spec: Reserved for future use.
    """

    batchable: bool = False
    batch_dim: t.Tuple[int, int] = (0, 0)
    # TODO: define input/output spec struct
    input_spec: t.Any = None
    output_spec: t.Any = None

    @classmethod
    def from_dict(cls, data: ModelSignatureDict) -> ModelSignature:
        if "batch_dim" in data and isinstance(data["batch_dim"], int):
            formated_data = dict(data, batch_dim=(data["batch_dim"], data["batch_dim"]))
        else:
            formated_data = data
        return bentoml_cattr.structure(formated_data, cls)

    @staticmethod
    def convert_signatures_dict(
        data: dict[str, ModelSignatureDict | ModelSignature]
    ) -> dict[str, ModelSignature]:
        return {
            k: ModelSignature.from_dict(v) if isinstance(v, dict) else v
            for k, v in data.items()
        }


if TYPE_CHECKING:
    ModelSignaturesType = dict[str, ModelSignature] | dict[str, ModelSignatureDict]


def model_signature_unstructure_hook(
    model_signature: ModelSignature,
) -> dict[str, t.Any]:
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


bentoml_cattr.register_unstructure_hook(
    ModelSignature, model_signature_unstructure_hook
)


@attr.define(repr=False, eq=False, frozen=True)
class ModelInfo:
    # for backward compatibility in case new fields are added to BentoInfo.
    __forbid_extra_keys__ = False
    # omit field in yaml file if it is not provided by the user.
    __omit_if_default__ = True

    tag: Tag
    name: str
    version: str
    module: str
    labels: t.Dict[str, str] = attr.field(
        validator=label_validator, converter=normalize_labels_value
    )
    _options: t.Dict[str, t.Any]
    metadata: MetadataDict = attr.field(validator=metadata_validator, converter=dict)
    context: ModelContext = attr.field()
    signatures: t.Dict[str, ModelSignature] = attr.field(
        converter=ModelSignature.convert_signatures_dict
    )
    api_version: str
    creation_time: datetime

    _cached_module: t.Optional[ModuleType] = None
    _cached_options: t.Optional[ModelOptions] = None

    def __init__(
        self,
        tag: Tag,
        module: str,
        labels: dict[str, str],
        options: dict[str, t.Any] | ModelOptions,
        metadata: MetadataDict,
        context: ModelContext,
        signatures: ModelSignaturesType,
        api_version: str,
        creation_time: datetime | None = None,
    ):
        if isinstance(options, ModelOptions):
            object.__setattr__(self, "_cached_options", options)
            options = options.to_dict()

        if creation_time is None:
            creation_time = datetime.now(timezone.utc)

        self.__attrs_init__(  # type: ignore
            tag=tag,
            name=tag.name,
            version=tag.version,
            module=module,
            labels=labels,
            options=options,
            metadata=metadata,
            context=context,
            signatures=signatures,
            api_version=api_version,
            creation_time=creation_time,
        )
        self.validate()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelInfo):
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

    # cached_property doesn't support __slots__ classes
    @property
    def imported_module(self) -> ModuleType:
        if self._cached_module is None:
            try:
                object.__setattr__(
                    self, "_cached_module", importlib.import_module(self.module)
                )
            except (ValueError, ModuleNotFoundError) as e:
                raise BentoMLException(
                    f"Module '{self.module}' defined in {MODEL_YAML_FILENAME} is not found."
                ) from e
        assert self._cached_module is not None
        return self._cached_module

    @property
    def options(self) -> ModelOptions:
        if self._cached_options is None:
            if hasattr(self.imported_module, "ModelOptions"):
                object.__setattr__(
                    self,
                    "_cached_options",
                    self.imported_module.ModelOptions(**self._options),
                )
            else:
                object.__setattr__(self, "_cached_options", ModelOptions())

        assert self._cached_options is not None
        return self._cached_options

    def to_dict(self) -> t.Dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)

    @overload
    def dump(self, stream: io.StringIO) -> io.BytesIO:
        ...

    @overload
    def dump(self, stream: None = None) -> None:
        ...

    def dump(self, stream: io.StringIO | None = None) -> io.BytesIO | None:
        return yaml.safe_dump(self.to_dict(), stream=stream, sort_keys=False)  # type: ignore (bad yaml types)

    @classmethod
    def from_yaml_file(cls, stream: t.IO[t.Any]) -> ModelInfo:
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:  # pragma: no cover - simple error handling
            logger.error(exc)
            raise

        if not isinstance(yaml_content, dict):
            raise BentoMLException(f"malformed {MODEL_YAML_FILENAME}")
        yaml_content["tag"] = str(
            Tag(t.cast(str, yaml_content["name"]), t.cast(str, yaml_content["version"]))
        )
        del yaml_content["name"]
        del yaml_content["version"]

        # For backwards compatibility for bentos created prior to version 1.0.0rc1
        if "bentoml_version" in yaml_content:
            del yaml_content["bentoml_version"]
        if "signatures" not in yaml_content:
            yaml_content["signatures"] = {}
        if "context" in yaml_content and "pip_dependencies" in yaml_content["context"]:
            del yaml_content["context"]["pip_dependencies"]
            yaml_content["context"]["framework_versions"] = {}

        try:
            model_info = bentoml_cattr.structure(yaml_content, cls)
        except TypeError as e:  # pragma: no cover - simple error handling
            raise BentoMLException(f"unexpected field in {MODEL_YAML_FILENAME}: {e}")
        return model_info

    def validate(self):
        # Validate model.yml file schema, content, bentoml version, etc
        # add tests when implemented
        ...


bentoml_cattr.register_structure_hook_func(
    lambda cls: issubclass(cls, ModelInfo),
    make_dict_structure_fn(
        ModelInfo,
        bentoml_cattr,
        name=override(omit=True),
        version=override(omit=True),
        _options=override(rename="options"),
    ),
)
bentoml_cattr.register_unstructure_hook_func(
    lambda cls: issubclass(cls, ModelInfo),
    # Ignore tag, tag is saved via the name and version field
    make_dict_unstructure_fn(
        ModelInfo,
        bentoml_cattr,
        tag=override(omit=True),
        _options=override(rename="options"),
        _cached_module=override(omit=True),
        _cached_options=override(omit=True),
    ),
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
