import os
import typing as t
import logging
from typing import TYPE_CHECKING
from datetime import datetime
from datetime import timezone

import fs
import attr
import yaml
import fs.osfs
import pathspec
import fs.errors
import fs.mirror
from fs.copy import copy_file
from cattr.gen import override
from cattr.gen import make_dict_unstructure_fn
from simple_di import inject
from simple_di import Provide

from ..tag import Tag
from ..store import Store
from ..store import StoreItem
from ..types import PathType
from ..utils import bentoml_cattr
from ..utils import copy_file_to_fs_folder
from ..models import ModelStore
from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException
from .build_config import BentoBuildConfig
from ..configuration import BENTOML_VERSION
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from fs.base import FS

    from bentoml import Runner

    from ..models import Model
    from ..service import Service
    from ..runner.runner import SimpleRunner
    from ..service.inference_api import InferenceAPI

logger = logging.getLogger(__name__)

BENTO_YAML_FILENAME = "bento.yaml"
BENTO_PROJECT_DIR_NAME = "src"
BENTO_README_FILENAME = "README.md"


def get_default_bento_readme(svc: "Service"):
    doc = f'# BentoML Service "{svc.name}"\n\n'
    doc += "This is a Machine Learning Service created with BentoML. \n\n"

    if svc.apis:
        doc += "## Inference APIs:\n\nIt contains the following inference APIs:\n\n"

        for api in svc.apis.values():
            doc += f"### /{api.name}\n\n"
            doc += f"* Input: {api.input.__class__.__name__}\n"
            doc += f"* Output: {api.output.__class__.__name__}\n\n"

    doc += """
## Customize This Message

This is the default generated `bentoml.Service` doc. You may customize it in your Bento
build file, e.g.:

```yaml
service: "image_classifier.py:svc"
description: "file: ./readme.md"
labels:
  foo: bar
  team: abc
docker:
  distro: debian
  gpu: True
python:
  packages:
    - tensorflow
    - numpy
```
"""
    # TODO: add links to documentation that may help with API client development
    return doc


@attr.define(repr=False, auto_attribs=False)
class Bento(StoreItem):
    _tag: Tag = attr.field()
    __fs: "FS" = attr.field()

    _info: "BentoInfo"

    _model_store: ModelStore
    _doc: t.Optional[str] = None

    _flushed: bool = False

    @staticmethod
    def _export_ext() -> str:
        return "bento"

    @__fs.validator  # type: ignore
    def check_fs(self, _attr: t.Any, new_fs: "FS"):
        try:
            new_fs.makedir("models", recreate=True)
        except fs.errors.ResourceReadOnly:
            # when we import a tarfile, it will be read-only, so just skip the step where we create
            # the models folder.
            pass
        self._model_store = ModelStore(new_fs.opendir("models"))

    def __init__(self, tag: Tag, bento_fs: "FS", info: "BentoInfo"):
        self._tag = tag
        self.__fs = bento_fs
        self.check_fs(None, bento_fs)
        self.info = info
        self.validate()

    @property
    def tag(self) -> Tag:
        return self._tag

    @property
    def _fs(self) -> "FS":
        return self.__fs

    @property
    def info(self) -> "BentoInfo":
        self._flushed = False
        return self._info

    @info.setter
    def info(self, new_info: "BentoInfo"):
        self._flushed = False
        self._info = new_info

    @classmethod
    @inject
    def create(
        cls,
        build_config: BentoBuildConfig,
        version: t.Optional[str] = None,
        build_ctx: t.Optional[str] = None,
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ) -> "Bento":
        from ..service.loader import import_service

        build_ctx = (
            os.getcwd()
            if build_ctx is None
            else os.path.realpath(os.path.expanduser(build_ctx))
        )
        assert os.path.isdir(build_ctx), f"build ctx {build_ctx} does not exist"

        # This also verifies that svc can be imported correctly
        svc = import_service(
            build_config.service, working_dir=build_ctx, change_global_cwd=True
        )

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

        models: "t.List[Model]" = []
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

        bento_fs.makedir("models", recreate=True)
        bento_model_store = ModelStore(bento_fs.opendir("models"))
        for model in models:
            logger.info('Packing model "%s" from "%s"', model.tag, model.path)
            model._save(bento_model_store)  # type: ignore[reportPrivateUsage]

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
                _path = fs.path.combine(dir_path, f.name).lstrip("/")
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
        if build_config.description is None:
            with bento_fs.open(BENTO_README_FILENAME, "w", encoding="utf-8") as f:
                f.write(get_default_bento_readme(svc))
        else:
            if build_config.description.startswith("file:"):
                file_name = build_config.description[5:].strip()
                copy_file_to_fs_folder(
                    file_name, bento_fs, dst_filename=BENTO_README_FILENAME
                )
            else:
                with bento_fs.open(BENTO_README_FILENAME, "w") as f:
                    f.write(build_config.description)

        # Create 'apis/openapi.yaml' file
        bento_fs.makedir("apis")
        with bento_fs.open(fs.path.combine("apis", "openapi.yaml"), "w") as f:
            yaml.dump(svc.openapi_doc(), f)

        res = Bento(
            tag,
            bento_fs,
            BentoInfo(
                tag=tag,
                service=svc,  # type: ignore # attrs converters do not typecheck
                labels=build_config.labels,
                models=[BentoModelInfo.from_bento_model(m) for m in models],
                runners=[BentoRunnerInfo.from_runner(r) for r in svc.runners.values()],
                apis=[
                    BentoApiInfo.from_inference_api(api) for api in svc.apis.values()
                ],
            ),
        )
        # Create bento.yaml
        res.flush_info()

        return res

    @classmethod
    def from_fs(cls, item_fs: "FS") -> "Bento":
        try:
            with item_fs.open(BENTO_YAML_FILENAME, "r", encoding="utf-8") as bento_yaml:
                info = BentoInfo.from_yaml_file(bento_yaml)
        except fs.errors.ResourceNotFound:
            raise BentoMLException(
                f"Failed to load bento because it does not contain a '{BENTO_YAML_FILENAME}'"
            )

        res = cls(info.tag, item_fs, info)
        if not res.validate():
            raise BentoMLException(
                f"Failed to create bento because it contains an invalid '{BENTO_YAML_FILENAME}'"
            )

        res._flushed = True
        return res

    @property
    def path(self) -> str:
        return self.path_of("/")

    def path_of(self, item: str) -> str:
        return self._fs.getsyspath(item)

    def flush_info(self):
        if self._flushed:
            return

        with self._fs.open(BENTO_YAML_FILENAME, "w") as bento_yaml:
            self.info.dump(bento_yaml)

        self._flushed = True

    @property
    def doc(self) -> str:
        if self._doc is not None:
            return self._doc

        with self._fs.open(BENTO_README_FILENAME, "r") as readme_md:
            self._doc = str(readme_md.read())
            return self._doc

    @property
    def creation_time(self) -> datetime:
        return self.info.creation_time

    @inject
    def save(
        self,
        bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
    ) -> "Bento":
        self.flush_info()

        if not self.validate():
            logger.warning(f"Failed to create Bento for {self.tag}, not saving.")
            raise BentoMLException("Failed to save Bento because it was invalid.")

        with bento_store.register(self.tag) as bento_path:
            out_fs = fs.open_fs(bento_path, create=True, writeable=True)
            fs.mirror.mirror(self._fs, out_fs, copy_if_newer=False)
            self._fs.close()
            self.__fs = out_fs

        return self

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
        return self._fs.isfile(BENTO_YAML_FILENAME)

    def __str__(self):
        return f'Bento(tag="{self.tag}")'


class BentoStore(Store[Bento]):
    def __init__(self, base_path: t.Union[PathType, "FS"]):
        super().__init__(base_path, Bento)


@attr.define
class BentoRunnerInfo:
    name: str
    runner_type: str

    @classmethod
    def from_runner(cls, r: "t.Union[Runner, SimpleRunner]") -> "BentoRunnerInfo":
        # Add runner default resource quota and batching config here
        return cls(
            name=r.name,  # type: ignore
            runner_type=r.__class__.__name__,
        )


@attr.define
class BentoApiInfo:
    name: str
    input_type: str
    output_type: str

    @classmethod
    def from_inference_api(cls, api: "InferenceAPI") -> "BentoApiInfo":
        return cls(
            name=api.name,
            input_type=api.input.__class__.__name__,
            output_type=api.output.__class__.__name__,
        )


@attr.define
class BentoModelInfo:
    tag: Tag = attr.field(converter=Tag.from_taglike)
    module: str
    creation_time: datetime

    @classmethod
    def from_bento_model(cls, bento_model: "Model") -> "BentoModelInfo":
        return cls(
            tag=bento_model.tag,
            module=bento_model.info.module,
            creation_time=bento_model.info.creation_time,
        )


@attr.define(repr=False, frozen=True)
class BentoInfo:
    tag: Tag
    service: str = attr.field(
        converter=lambda svc: svc if isinstance(svc, str) else svc._import_str
    )  # type: ignore[reportPrivateUsage]
    name: str = attr.field(init=False)  # converted from tag in __attrs_post_init__
    version: str = attr.field(init=False)  # converted from tag in __attrs_post_init__
    bentoml_version: str = attr.field(default=BENTOML_VERSION)
    creation_time: datetime = attr.field(factory=lambda: datetime.now(timezone.utc))

    labels: t.Dict[str, t.Any] = attr.field(factory=dict)
    models: t.List[BentoModelInfo] = attr.field(factory=list)
    runners: t.List[BentoRunnerInfo] = attr.field(factory=list)
    apis: t.List[BentoApiInfo] = attr.field(factory=list)

    _flushed: bool = False

    def __attrs_post_init__(self):
        # Direct set is not available when frozen=True
        object.__setattr__(self, "name", self.tag.name)
        object.__setattr__(self, "version", self.tag.version)

        self.validate()

    def to_dict(self) -> t.Dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)

    def dump(self, stream: t.IO[t.Any]):
        return yaml.dump(self, stream, sort_keys=False)

    @classmethod
    def from_yaml_file(cls, stream: t.IO[t.Any]) -> "BentoInfo":
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise

        assert yaml_content is not None

        yaml_content["tag"] = Tag(yaml_content["name"], yaml_content["version"])
        del yaml_content["name"]
        del yaml_content["version"]

        if "models" in yaml_content:
            # For backwards compatibility for bentos created prior to version 1.0.0a7
            models = yaml_content["models"]
            if models and len(models) > 0 and isinstance(models[0], str):
                yaml_content["models"] = list(
                    map(
                        lambda model_tag: {
                            "tag": model_tag,
                            "module": "unknown",
                            "creation_time": datetime.fromordinal(1),
                        },
                        models,
                    )
                )
        try:
            # type: ignore[attr-defined]
            return bentoml_cattr.structure(yaml_content, cls)
        except KeyError as e:
            raise BentoMLException(f"Missing field {e} in {BENTO_YAML_FILENAME}")

    def validate(self):
        # Validate bento.yml file schema, content, bentoml version, etc
        ...


bentoml_cattr.register_unstructure_hook(
    BentoInfo,
    # Ignore internal private state "_flushed"
    # Ignore tag, tag is saved via the name and version field
    make_dict_unstructure_fn(
        BentoInfo, bentoml_cattr, _flushed=override(omit=True), tag=override(omit=True)
    ),
)


def _BentoInfo_dumper(dumper: yaml.Dumper, info: BentoInfo) -> yaml.Node:
    return dumper.represent_dict(info.to_dict())


yaml.add_representer(BentoInfo, _BentoInfo_dumper)
