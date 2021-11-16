import logging
import os
import typing as t
from datetime import datetime, timezone

import attr
import fs
import fs.errors
import fs.mirror
import fs.osfs
import pathspec
import yaml
from fs.base import FS
from fs.copy import copy_dir, copy_file
from simple_di import Provide, inject

from bentoml import __version__

from ...exceptions import BentoMLException, InvalidArgument
from ..configuration.containers import BentoMLContainer
from ..store import Store, StoreItem
from ..types import PathType, Tag
from ..utils import cached_property
from .env import BentoEnv

if t.TYPE_CHECKING:
    from ..models.store import ModelInfo, ModelStore
    from ..service import Service
    from ..service.inference_api import InferenceAPI

logger = logging.getLogger(__name__)

BENTO_YAML_FILENAME = "bento.yaml"
BENTO_PROJECT_DIR_NAME = "src"


@attr.define(repr=False)
class Bento(StoreItem):
    _tag: Tag
    _fs: FS

    @property
    def tag(self) -> Tag:
        return self._tag

    @staticmethod
    @inject
    def create(
        svc: "Service",
        build_ctx: PathType,
        additional_models: t.List[str],
        version: t.Optional[str],
        include: t.List[str],
        exclude: t.List[str],
        env: t.Dict[str, t.Any],
        labels: t.Dict[str, str],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ) -> "OSBento":
        tag = Tag(svc.name, version)
        if version is None:
            tag = tag.make_new_version()

        logger.debug(f"Building BentoML service {tag} from build context {build_ctx}")

        bento_fs = fs.open_fs(f"temp://bentoml_bento_{svc.name}")
        if isinstance(build_ctx, os.PathLike):
            build_ctx = build_ctx.__fspath__()
        ctx_fs = fs.open_fs(build_ctx)

        models = additional_models
        # Add Runner required models to models list
        for runner in svc._runners.values():  # type: ignore[reportPrivateUsage]
            models += runner.required_models

        models_list: "list[ModelInfo]" = []
        seen_model_tags = []
        for model_tag in models:
            try:
                model_info = model_store.get(model_tag)
                if model_info.tag not in seen_model_tags:
                    seen_model_tags.append(model_info.tag)
                    models_list.append(model_info)
            except FileNotFoundError:
                raise BentoMLException(
                    f"Model {model_tag} not found in local model store"
                )

        for model in models_list:
            model_name, model_version = model.tag.split(":")
            target_path = os.path.join("models", model_name, model_version)
            bento_fs.makedirs(target_path, recreate=True)
            copy_dir(fs.open_fs(model.path), "/", bento_fs, target_path)

        # Copy all files base on include and exclude, into `{svc.name}` directory
        relpaths = [s for s in include if s.startswith("../")]
        if len(relpaths) != 0:
            raise InvalidArgument(
                "Paths outside of the build context directory cannot be included; use a symlink or copy those files into the working directory manually."
            )

        spec = pathspec.PathSpec.from_lines("gitwildmatch", include)
        exclude_spec = pathspec.PathSpec.from_lines("gitwildmatch", exclude)
        exclude_specs: "list[t.Tuple[str, pathspec.PathSpec]]" = []
        bento_fs.makedir(BENTO_PROJECT_DIR_NAME)
        target_fs = bento_fs.opendir(BENTO_PROJECT_DIR_NAME)

        for dir_path, _, files in ctx_fs.walk():
            for ignore_file in [f for f in files if f.name == ".bentoignore"]:
                exclude_specs.append(
                    (
                        dir_path,
                        pathspec.PathSpec.from_lines(
                            "gitwildmatch", ctx_fs.open(ignore_file.make_path(dir_path))
                        ),
                    )
                )

            cur_exclude_specs: list[t.Tuple[str, pathspec.PathSpec]] = []
            for ignore_path, _spec in exclude_specs:
                if fs.path.isparent(ignore_path, dir_path):
                    cur_exclude_specs.append((ignore_path, _spec))

            for f in files:
                _path = fs.path.combine(dir_path, f.name)
                if spec.match_file(_path) and not exclude_spec.match_file(_path):
                    if not any(
                        [
                            _spec.match_file(fs.path.relativefrom(ignore_path, _path))
                            for ignore_path, _spec in cur_exclude_specs
                        ]
                    ):
                        target_fs.makedirs(dir_path, recreate=True)
                        copy_file(ctx_fs, _path, target_fs, _path)

        # Create env, docker, bentoml dev whl files
        bento_env = BentoEnv(build_ctx=build_ctx, **env)
        bento_env.save(bento_fs)

        # Create `readme.md` file
        with bento_fs.open("README.md", "w") as f:
            f.write(svc.doc)

        # Create 'apis/openapi.yaml' file
        bento_fs.makedir("apis")
        with bento_fs.open(fs.path.combine("apis", "openapi.yaml"), "w") as f:
            yaml.dump(svc.openapi_doc(), f)

        # Create bento.yaml
        with bento_fs.open(BENTO_YAML_FILENAME, "w") as bento_yaml:
            BentoInfo.from_build_args(tag, svc, labels, models).dump(bento_yaml)

        return OSBento(tag, bento_fs)

    @classmethod
    def from_fs(cls, tag: Tag, fs: FS) -> "Bento":
        res = cls(Tag("", ""), fs)
        res._tag = res.info.tag

        # TODO: Check bento_metadata['bentoml_version'] and show user warning if needed
        return res

    @property
    def path(self) -> t.Optional[str]:
        try:
            return OSBento.from_Bento(self).path
        except fs.errors.NoSysPath:
            return None

    @cached_property
    def info(self) -> "BentoInfo":
        with self._fs.open(BENTO_YAML_FILENAME, "r") as bento_yaml:
            return BentoInfo.from_yaml_file(bento_yaml)

    @property
    def creation_time(self) -> datetime:
        return self.info.creation_time

    def save(
        self, bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store]
    ) -> "OSBento":
        try:
            OSBento.from_Bento(self).save(bento_store)
        except TypeError:
            pass

        if not self.validate():
            logger.warning(f"Failed to create Bento for {self.tag}, not saving.")
            raise BentoMLException("Failed to save Bento because it was invalid")

        with bento_store.register(self.tag) as bento_path:
            out_fs = fs.open_fs(bento_path, create=True, writeable=True)
            fs.mirror.mirror(self._fs, out_fs, copy_if_newer=False)
            self._fs.close()
            self._fs = out_fs

        return OSBento.from_Bento(self)

    def export(self, path: str):
        out_fs = fs.open_fs(path, create=True, writeable=True)
        fs.mirror.mirror(self._fs, out_fs, copy_if_newer=False)

    def push(self):
        pass

    def validate(self):
        return self._fs.isfile(BENTO_YAML_FILENAME)

    def __str__(self):
        return f'Bento(tag="{self.tag}", path="{self.path}")'


class OSBento(Bento):
    @staticmethod
    def from_Bento(bento: Bento) -> "OSBento":
        if not isinstance(bento._fs, fs.osfs.OSFS):
            raise TypeError(f"this Bento ('{bento._tag}') is not in the OS filesystem")
        bento.__class__ = OSBento
        return bento  # type: ignore

    @property
    def path(self) -> str:
        return self._fs.getsyspath("/")

    def save(
        self, bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store]
    ) -> "OSBento":
        if not self.validate():
            logger.warning(f"Failed to create Bento for {self.tag}, not saving.")
            raise BentoMLException("Failed to save Bento because it was invalid")

        with bento_store.register(self.tag) as bento_path:
            os.rmdir(bento_path)
            os.rename(self._fs.getsyspath("/"), bento_path)
            self._fs.close()
            self._fs = fs.open_fs(bento_path)

        return self


class BentoStore(Store[OSBento]):
    def __init__(self, base_path: PathType):
        super().__init__(base_path, OSBento)


@attr.define(repr=False)
class BentoInfo:
    service: str
    tag: Tag
    bentoml_version: str
    creation_time: datetime
    labels: t.Dict[str, t.Any]
    apis: t.Dict[str, InferenceAPI]
    models: t.List[str]

    def dump(self, stream: t.IO[t.Any]):
        return yaml.dump(self, stream, sort_keys=False)

    @classmethod
    def from_build_args(
        cls, tag: Tag, svc: "Service", labels: t.Dict[str, t.Any], models: t.List[str]
    ):
        bento_info = cls(
            service=svc._import_str,  # type: ignore[reportPrivateUsage]
            tag=tag,
            bentoml_version=__version__,
            creation_time=datetime.now(timezone.utc),
            labels=labels,  # TODO: validate user provided labels
            apis=svc._apis,  # type: ignore[reportPrivateUsage]
            models=models,  # TODO: populate with model & framework info
        )

        bento_info.validate()
        return bento_info

    @classmethod
    def from_yaml_file(cls, stream: t.IO[t.Any]):
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise

        bento_info = cls(**yaml_content)
        bento_info.validate()
        return bento_info

    def validate(self):
        # Validate bento.yml file schema, content, bentoml version, etc
        ...


def _BentoInfo_dumper(dumper: yaml.Dumper, info: BentoInfo) -> yaml.Node:
    to_dump = attr.asdict(info)
    to_dump["name"] = info.tag.name
    to_dump["version"] = info.tag.version
    del to_dump["tag"]
    return dumper.represent_dict(to_dump)


yaml.add_representer(BentoInfo, _BentoInfo_dumper)
