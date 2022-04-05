import typing as t
import logging
from sys import version_info as pyver
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone

import fs
import attr
import yaml
import fs.errors
import fs.mirror
import cloudpickle
from fs.base import FS
from simple_di import inject
from simple_di import Provide

from ..tag import Tag
from ..store import Store
from ..store import StoreItem
from ...exceptions import NotFound
from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..types import PathType
    from ..runner import Runner

logger = logging.getLogger(__name__)

PYTHON_VERSION: str = f"{pyver.major}.{pyver.minor}.{pyver.micro}"
MODEL_YAML_FILENAME = "model.yaml"
CUSTOM_OBJECTS_FILENAME = "custom_objects.pkl"


@attr.define(repr=False)
class Model(StoreItem):
    _tag: Tag
    __fs: FS

    _info: "ModelInfo"
    _custom_objects: t.Optional[t.Dict[str, t.Any]] = None

    _info_flushed = False
    _custom_objects_flushed = False

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
        self._info_flushed = False
        return self._info

    @info.setter
    def info(self, new_info: "ModelInfo"):
        self._info_flushed = False
        self._info = new_info

    @property
    def custom_objects(self) -> t.Dict[str, t.Any]:
        self._custom_objects_flushed = False

        if self._custom_objects is None:
            if self._fs.isfile(CUSTOM_OBJECTS_FILENAME):
                with self._fs.open(CUSTOM_OBJECTS_FILENAME, "rb") as cofile:
                    self._custom_objects = cloudpickle.load(cofile)
            else:
                self._custom_objects = {}

        return self._custom_objects

    def __eq__(self, other: "Model") -> bool:
        return self._tag == other._tag

    def __hash__(self) -> int:
        return hash(self._tag)

    @staticmethod
    def create(
        name: str,
        *,
        module: str = "",
        labels: t.Optional[t.Dict[str, str]] = None,
        options: t.Optional[t.Dict[str, t.Any]] = None,
        custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        context: t.Optional[t.Dict[str, t.Any]] = None,
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
        options = {} if options is None else options
        metadata = {} if metadata is None else metadata
        context = {} if context is None else context
        context["bentoml_version"] = BENTOML_VERSION
        context["python_version"] = PYTHON_VERSION

        model_fs = fs.open_fs(f"temp://bentoml_model_{name}")

        res = Model(
            tag,
            model_fs,
            ModelInfo(
                tag=tag,
                module=module,
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
        self, model_store: "ModelStore" = Provide[BentoMLContainer.model_store]
    ) -> "Model":
        self._save(model_store)

        return self

    def _save(self, model_store: "ModelStore") -> "Model":
        self.flush_info()
        self.flush_custom_objects()

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

        res._info_flushed = True
        res._custom_objects_flushed = True
        return res

    @property
    def path(self) -> str:
        return self.path_of("/")

    def path_of(self, item: str) -> str:
        return self._fs.getsyspath(item)

    def flush_info(self):
        if self._info_flushed:
            return

        with self._fs.open(MODEL_YAML_FILENAME, "w") as model_yaml:
            self.info.dump(model_yaml)

        self._info_flushed = True

    def flush_custom_objects(self):
        if self._custom_objects_flushed:
            return

        # pickle custom_objects if it is not None and not empty
        if self.custom_objects:
            with self._fs.open(CUSTOM_OBJECTS_FILENAME, "wb") as cofile:
                cloudpickle.dump(self.custom_objects, cofile)

        self._custom_objects_flushed = True

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
        self.flush_info()
        self.flush_custom_objects()
        return super().export(
            path,
            output_format,
            protocol=protocol,
            user=user,
            passwd=passwd,
            params=params,
            subpath=subpath,
        )

    def load_runner(self) -> "Runner":
        raise NotImplementedError

    def validate(self):
        return self._fs.isfile(MODEL_YAML_FILENAME)

    def __str__(self):
        return f'Model(tag="{self.tag}", path="{self.path}")'


class ModelStore(Store[Model]):
    def __init__(self, base_path: "t.Union[PathType, FS]"):
        super().__init__(base_path, Model)


@attr.define(repr=False)
class ModelInfo:
    tag: Tag
    module: str
    labels: t.Dict[str, t.Any]
    options: t.Dict[str, t.Any]
    metadata: t.Dict[str, t.Any]
    context: t.Dict[str, t.Any]
    bentoml_version: str = BENTOML_VERSION
    api_version: str = "v1"
    creation_time: datetime = attr.field(factory=lambda: datetime.now(timezone.utc))

    def __attrs_post_init__(self):
        self.validate()

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "name": self.tag.name,
            "version": self.tag.version,
            "bentoml_version": self.bentoml_version,
            "creation_time": self.creation_time,
            "api_version": self.api_version,
            "module": self.module,
            "context": self.context,
            "labels": self.labels,
            "options": self.options,
            "metadata": self.metadata,
        }

    def dump(self, stream: t.IO[t.Any]):
        return yaml.dump(self, stream, sort_keys=False)

    @classmethod
    def from_yaml_file(cls, stream: t.IO[t.Any]):
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
            model_info = cls(**yaml_content)  # type: ignore
        except TypeError:  # pragma: no cover - simple error handling
            raise BentoMLException(f"unexpected field in {MODEL_YAML_FILENAME}")
        return model_info

    def validate(self):
        # Validate model.yml file schema, content, bentoml version, etc
        # add tests when implemented
        ...


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
