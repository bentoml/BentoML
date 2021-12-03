import logging
import os
import shutil
import typing as t
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import attr
import fs
import fs.errors
import fs.mirror
import fs.osfs
import pathspec
import yaml
from fs.copy import copy_file
from simple_di import Provide, inject

from ...exceptions import BentoMLException, InvalidArgument
from ..configuration import BENTOML_VERSION
from ..configuration.containers import BentoMLContainer
from ..models import ModelStore
from ..store import Store, StoreItem
from ..types import PathType, Tag
from .build_config import BentoBuildConfig

if TYPE_CHECKING:
    from fs.base import FS

    from ..models import Model
    from ..service import Service

logger = logging.getLogger(__name__)

BENTO_YAML_FILENAME = "bento.yaml"
BENTO_PROJECT_DIR_NAME = "src"
BENTO_READEME_FILENAME = "README.md"


def _get_default_bento_readme(svc: "Service"):
    doc = f'# BentoML Service "{svc.name}"\n\n'
    doc += "This is a Machine Learning Service created with BentoML. \n\n"

    if svc._apis:  # type: ignore[reportPrivateUsage]
        doc += "## Inference APIs:\n\nIt contains the following inference APIs:\n\n"

        for api in svc._apis.values():  # type: ignore[reportPrivateUsage]
            doc += f"### /{api.name}\n\n"
            doc += f"* Input: {api.input.__class__.__name__}\n"
            doc += f"* Output: {api.output.__class__.__name__}\n\n"

    doc += """
## Customize This Message

This is the default generated `bentoml.Service` doc. You may customize it in your Bento
build file:

```yaml
service: "image_classifier.py:svc"
description: "./readme.md"
labels:
  foo: bar
  team: abc
docker:
  distro: slim
  gpu: True
python:
  packages:
    - tensorflow
    - numpy
```
"""
    # TODO: add links to documentation that may help with API client development
    return doc


def _Bento_set_fs(bento: "Bento", _: t.Any, new_fs: "FS") -> "FS":
    new_fs.makedir("models", recreate=True)
    bento._model_store = ModelStore(new_fs.opendir("models"))  # type: ignore[reportPrivateUsage]
    return new_fs


@attr.define(repr=False, auto_attribs=False)
class Bento(StoreItem):
    _tag: Tag = attr.field()
    _fs: "FS" = attr.field(on_setattr=_Bento_set_fs)

    _model_store: ModelStore
    _info: t.Optional["BentoInfo"] = None
    _doc: t.Optional[str] = None

    def __init__(self, tag: Tag, bento_fs: "FS"):
        self._tag = tag
        self._fs = _Bento_set_fs(self, None, bento_fs)
        self.validate()

    @property
    def tag(self) -> Tag:
        return self._tag

    @staticmethod
    @inject
    def create(
        build_config: BentoBuildConfig,
        version: t.Optional[str] = None,
        build_ctx: t.Optional[str] = None,
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ) -> "SysPathBento":
        from ..service.loader import import_service

        build_ctx = (
            os.getcwd()
            if build_ctx is None
            else os.path.realpath(os.path.expanduser(build_ctx))
        )
        assert os.path.isdir(build_ctx), f"build ctx {build_ctx} does not exist"

        # This also verifies that svc can be imported correctly
        svc = import_service(build_config.service, working_dir=build_ctx)

        # Apply default build options
        build_config = build_config.with_defaults()

        tag = Tag(svc.name, version)
        if version is None:
            tag = tag.make_new_version()

        logger.info(
            f'Building BentoML service "{tag}" from build context "{build_ctx}"'
        )

        bento_fs = fs.open_fs(f"temp://bentoml_bento_{svc.name}")
        ctx_fs = fs.open_fs(build_ctx)

        model_tags = build_config.additional_models
        # Add Runner required models to models list
        for runner in svc.runners.values():
            model_tags += runner.required_models

        models: t.List[Model] = []
        seen_model_tags: t.Set[Tag] = set()
        for model_tag in model_tags:
            try:
                model_info = model_store.get(model_tag)
                if model_info.tag not in seen_model_tags:
                    seen_model_tags.add(model_info.tag)
                    models.append(model_info)
            except FileNotFoundError:
                raise BentoMLException(
                    f"Model {model_tag} not found in local model store"
                )

        logger.info(
            "Packing required models: %s",
            ", ".join(map(lambda m: f'"{m.tag}"', models)),
        )
        bento_fs.makedir("models", recreate=True)
        bento_model_store = ModelStore(bento_fs.opendir("models"))
        for model in models:
            model.save(bento_model_store)

        # Copy all files base on include and exclude, into `{svc.name}` directory
        relpaths = [s for s in build_config.include if s.startswith("../")]
        if len(relpaths) != 0:
            raise InvalidArgument(
                "Paths outside of the build context directory cannot be included; use a symlink or copy those files into the working directory manually."
            )

        spec = pathspec.PathSpec.from_lines("gitwildmatch", build_config.include)
        exclude_spec = pathspec.PathSpec.from_lines(
            "gitwildmatch", build_config.exclude
        )
        exclude_specs: t.List[t.Tuple[str, pathspec.PathSpec]] = []
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

            cur_exclude_specs: t.List[t.Tuple[str, pathspec.PathSpec]] = []
            for ignore_path, _spec in exclude_specs:
                if fs.path.isparent(ignore_path, dir_path):
                    cur_exclude_specs.append((ignore_path, _spec))

            for f in files:
                _path = fs.path.combine(dir_path, f.name)
                if spec.match_file(_path) and not exclude_spec.match_file(_path):
                    if not any(
                        _spec.match_file(fs.path.relativefrom(ignore_path, _path))
                        for ignore_path, _spec in cur_exclude_specs
                    ):
                        target_fs.makedirs(dir_path, recreate=True)
                        copy_file(ctx_fs, _path, target_fs, _path)

        if build_config.docker:
            build_config.docker.write_to_bento(bento_fs, build_ctx)
        if build_config.python:
            build_config.python.write_to_bento(bento_fs, build_ctx)
        if build_config.conda:
            build_config.conda.write_to_bento(bento_fs, build_ctx)

        # Create `readme.md` file
        with bento_fs.open(BENTO_READEME_FILENAME, "w") as f:
            if build_config.description is None:
                f.write(_get_default_bento_readme(svc))
            else:
                f.write(build_config.description)

        # Create 'apis/openapi.yaml' file
        bento_fs.makedir("apis")
        with bento_fs.open(fs.path.combine("apis", "openapi.yaml"), "w") as f:
            yaml.dump(svc.openapi_doc(), f)

        # Create bento.yaml
        with bento_fs.open(BENTO_YAML_FILENAME, "w") as bento_yaml:
            # pyright doesn't know about attrs converters
            BentoInfo(  # type: ignore
                tag, svc, build_config.labels, [str(tag) for tag in seen_model_tags]
            ).dump(bento_yaml)

        return SysPathBento(tag, bento_fs)

    @classmethod
    def from_fs(cls, item_fs: "FS") -> "Bento":
        res = cls(None, item_fs)  # type: ignore
        res._tag = res.info.tag

        # TODO: Check bento_metadata['bentoml_version'] and show user warning if needed
        return res

    @property
    def path(self) -> t.Optional[str]:
        try:
            return SysPathBento.from_Bento(self).path
        except TypeError:
            return None

    @property
    def info(self) -> "BentoInfo":
        if self._info is not None:
            return self._info

        with self._fs.open(BENTO_YAML_FILENAME, "r") as bento_yaml:
            self._info = BentoInfo.from_yaml_file(bento_yaml)
            return self._info

    @property
    def doc(self) -> str:
        if self._doc is not None:
            return self._doc

        with self._fs.open(BENTO_READEME_FILENAME, "r") as readme_md:
            self._doc = str(readme_md.read())
            return self._doc

    @property
    def creation_time(self) -> datetime:
        return self.info.creation_time

    def save(
        self, bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store]
    ) -> "SysPathBento":
        try:
            SysPathBento.from_Bento(self).save(bento_store)
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

        return SysPathBento.from_Bento(self)

    def export(self, path: str):
        out_fs = fs.open_fs(path, create=True, writeable=True)
        fs.mirror.mirror(self._fs, out_fs, copy_if_newer=False)

    def push(self):
        pass

    def validate(self):
        return self._fs.isfile(BENTO_YAML_FILENAME)

    def __str__(self):
        return f'Bento(tag="{self.tag}", path="{self.path}")'


class SysPathBento(Bento):
    @staticmethod
    def from_Bento(bento: Bento) -> "SysPathBento":
        try:
            bento._fs.getsyspath("/")
        except fs.errors.NoSysPath:
            raise TypeError(f"this Bento ('{bento._tag}') is not in the OS filesystem")
        bento.__class__ = SysPathBento
        return bento  # type: ignore

    @property
    def path(self) -> str:
        return self._fs.getsyspath("/")

    def save(
        self, bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store]
    ) -> "SysPathBento":
        if not self.validate():
            logger.warning(f"Failed to create Bento for {self.tag}, not saving.")
            raise BentoMLException("Failed to save Bento because it was invalid")

        with bento_store.register(self.tag) as bento_path:
            os.rmdir(bento_path)
            shutil.move(self._fs.getsyspath("/"), bento_path)
            self._fs.close()
            self._fs = fs.open_fs(bento_path)

        return self


class BentoStore(Store[SysPathBento]):
    def __init__(self, base_path: t.Union[PathType, "FS"]):
        super().__init__(base_path, SysPathBento)


@attr.define(repr=False, frozen=True)
class BentoInfo:
    tag: Tag
    service: str = attr.field(
        converter=lambda svc: svc if isinstance(svc, str) else svc._import_str
    )  # type: ignore[reportPrivateUsage]
    labels: t.Dict[str, t.Any]  # TODO: validate user-provide labels
    models: t.List[str]  # TODO: populate with model & framework info
    bentoml_version: str = BENTOML_VERSION
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
            bento_info = cls(**yaml_content)
        except TypeError:
            raise BentoMLException(f"unexpected field in {BENTO_YAML_FILENAME}")

        return bento_info

    def validate(self):
        # Validate bento.yml file schema, content, bentoml version, etc
        ...


def _BentoInfo_dumper(dumper: yaml.Dumper, info: BentoInfo) -> yaml.Node:
    return dumper.represent_dict(
        {
            "service": info.service,
            "name": info.tag.name,
            "version": info.tag.version,
            "bentoml_version": info.bentoml_version,
            "creation_time": info.creation_time,
            "labels": info.labels,
            "models": info.models,
        }
    )


yaml.add_representer(BentoInfo, _BentoInfo_dumper)
