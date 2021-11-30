import logging
import typing as t
from collections import UserDict
from datetime import datetime, timezone

import attr
import fs
import fs.errors
import fs.mirror
import yaml
from fs.base import FS
from simple_di import Provide, inject

from ...exceptions import BentoMLException
from ..configuration import BENTOML_VERSION
from ..configuration.containers import BentoMLContainer
from ..store import Store, StoreItem
from ..types import PathType, Tag

if t.TYPE_CHECKING:
    from ..runner import Runner

logger = logging.getLogger(__name__)

MODEL_YAML_FILENAME = "model.yaml"


@attr.define(repr=False)
class Model(StoreItem):
    _tag: Tag
    _fs: FS

    _info: t.Optional["ModelInfo"] = None

    @property
    def tag(self) -> Tag:
        return self._tag

    def __eq__(self, other: "Model") -> bool:
        return self._tag == other._tag

    def __hash__(self) -> int:
        return hash(self._tag)

    @staticmethod
    def create(
        name: str,
        *,
        module: str = "",
        labels: t.Optional[t.Dict[str, t.Any]] = None,
        options: t.Optional[t.Dict[str, t.Any]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        framework_context: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> "SysPathModel":
        tag = Tag(name).make_new_version()
        labels = {} if labels is None else labels
        options = {} if options is None else options
        metadata = {} if metadata is None else metadata
        framework_context = {} if framework_context is None else framework_context

        model_fs = fs.open_fs(f"temp://bentoml_model_{name}")

        res = SysPathModel(tag, model_fs)
        res._info = ModelInfo(
            tag=tag,
            module=module,
            labels=labels,
            options=options,
            metadata=metadata,
            context=framework_context,
        )

        return res

    @inject
    def save(
        self, model_store: "ModelStore" = Provide[BentoMLContainer.model_store]
    ) -> "SysPathModel":
        if self._info is not None:
            with self._fs.open(MODEL_YAML_FILENAME, "w") as model_yaml:
                self._info.dump(model_yaml)

        if not self.validate():
            logger.warning(f"Failed to create Model for {self.tag}, not saving.")
            raise BentoMLException("Failed to save Model because it was invalid")

        with model_store.register(self.tag) as model_path:
            out_fs = fs.open_fs(model_path, create=True, writeable=True)
            fs.mirror.mirror(self._fs, out_fs, copy_if_newer=False)
            self._fs.close()
            self._fs = out_fs

        return SysPathModel.from_Model(self)

    @classmethod
    def from_fs(cls, item_fs: FS) -> "Model":
        res = cls(None, item_fs)  # type: ignore
        res._tag = res.info.tag

        return res

    @property
    def path(self) -> t.Optional[str]:
        return self.path_of("/")

    def path_of(self, item: str) -> t.Optional[str]:
        try:
            return SysPathModel.from_Model(self).path_of(item)
        except TypeError:
            return None

    @property
    def info(self) -> "ModelInfo":
        if self._info is not None:
            return self._info

        with self._fs.open(MODEL_YAML_FILENAME, "r") as model_yaml:
            self._info = ModelInfo.from_yaml_file(model_yaml)
            return self._info

    @property
    def creation_time(self) -> datetime:
        return self.info["creation_time"]

    def export(self, path: str):
        out_fs = fs.open_fs(path, create=True, writeable=True)
        fs.mirror.mirror(self._fs, out_fs, copy_if_newer=False)

    def push(self):
        pass

    def load_runner(self) -> "Runner":
        raise NotImplementedError

    def validate(self):
        return self._fs.isfile(MODEL_YAML_FILENAME)

    def __str__(self):
        return f'Bento(tag="{self.tag}", path="{self.path}")'


class SysPathModel(Model):
    @staticmethod
    def from_Model(model: Model) -> "SysPathModel":
        try:
            model._fs.getsyspath("/")
        except fs.errors.NoSysPath:
            raise TypeError(f"{model} is not in the OS filesystem")
        model.__class__ = SysPathModel
        return model  # type: ignore

    @property
    def path(self) -> str:
        return self.path_of("/")

    def path_of(self, item: str) -> str:
        return self._fs.getsyspath(item)


class ModelStore(Store[SysPathModel]):
    def __init__(self, base_path: t.Union[PathType, FS]):
        super().__init__(base_path, SysPathModel)


if t.TYPE_CHECKING:
    ModelInfoBase = UserDict[str, t.Any]
else:
    ModelInfoBase = UserDict


@attr.define(repr=False)
class ModelInfo(ModelInfoBase):
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

    def dump(self, stream: t.IO[t.Any]):
        return yaml.dump(self, stream, sort_keys=False)

    @classmethod
    def from_yaml_file(cls, stream: t.IO[t.Any]):
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise

        yaml_content["tag"] = Tag(yaml_content["name"], yaml_content["version"])
        del yaml_content["name"]
        del yaml_content["version"]

        try:
            model_info = cls(**yaml_content)
        except TypeError:
            raise BentoMLException(f"unexpected field in {MODEL_YAML_FILENAME}")
        return model_info

    def validate(self):
        # Validate model.yml file schema, content, bentoml version, etc
        ...


def _ModelInfo_dumper(dumper: yaml.Dumper, info: ModelInfo) -> yaml.Node:
    return dumper.represent_dict(
        {
            "name": info.tag.name,
            "version": info.tag.version,
            "bentoml_version": info.bentoml_version,
            "creation_time": info.creation_time,
            "api_version": info.api_version,
            "module": info.module,
            "context": info.context,
            "labels": info.labels,
            "options": info.options,
            "metadata": info.metadata,
        },
    )


yaml.add_representer(ModelInfo, _ModelInfo_dumper)
