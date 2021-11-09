import datetime
import logging
import os
import typing as t
from collections import OrderedDict, UserDict

import attr
import fs
import pathspec
import yaml
from fs.base import FS
from fs.copy import copy_dir, copy_file
from fs.mirror import mirror
from simple_di import Provide, inject

from bentoml import __version__

from ...exceptions import BentoMLException, InvalidArgument
from ..configuration.containers import BentoMLContainer
from ..store import Store, StoreItem
from ..types import PathType, Tag
from ..utils import cached_property
from .env import BentoEnv

if t.TYPE_CHECKING:
    from ..models.store import ModelStore
    from ..service import Service

logger = logging.getLogger(__name__)


BENTO_YAML_FILENAME = "bento.yaml"


@attr.define(repr=False)
class Bento(StoreItem):
    tag: Tag
    fs: FS

    @staticmethod
    @inject
    def create(
        svc: "Service",
        build_ctx: PathType,
        models: t.List[str],
        version: t.Optional[str],
        description: t.Optional[str],
        include: t.List[str],
        exclude: t.List[str],
        env: t.Optional[t.Dict[str, t.Any]],
        labels: t.Optional[t.Dict[str, str]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ) -> "Bento":
        tag = Tag(svc.name, version)

        logger.debug(f"Building BentoML service {tag} from build context {build_ctx}")

        bento_fs = fs.open_fs(f"temp://bentoml_bento_{svc.name}")
        ctx_fs = fs.open_fs(build_ctx)

        # Add Runner required models if models is not specified
        for runner in svc._runners.values():
            models += runner.required_models

        models_list = []
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
        exclude_specs = []
        bento_fs.makedir(svc.name)
        target_fs = bento_fs.opendir(svc.name)

        for dir_path, _, files in ctx_fs.walk():
            for ignore_file in [f for f in files if f.name == ".bentomlignore"]:
                exclude_specs.append(
                    (
                        dir_path,
                        pathspec.PathSpec.from_lines(
                            "gitwildmatch", ctx_fs.open(ignore_file.make_path(dir_path))
                        ),
                    )
                )

            cur_exclude_specs = []
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
            f.write(description)

        # Create 'apis/openapi.yaml' file
        bento_fs.makedir("apis")
        with bento_fs.open(fs.path.combine("apis", "openapi.yaml"), "w") as f:
            # Add representer to allow dumping OrderedDict as a regular map
            yaml.add_representer(
                OrderedDict,
                lambda dumper, data: dumper.represent_mapping(
                    "tag:yaml.org,2002:map", data.items()
                ),
            )
            yaml.dump(svc.openapi_doc(), f)

        # Create bento.yaml
        with bento_fs.open(BENTO_YAML_FILENAME, "w") as bento_yaml:
            BentoMetadata.from_build_args(tag, svc, labels, models).dump(bento_yaml)

        return Bento(tag, bento_fs)

    @classmethod
    def from_fs(cls, tag: Tag, bento_fs: FS) -> "Bento":
        with bento_fs.open(BENTO_YAML_FILENAME, "r") as bento_yaml:
            bento_metadata = BentoMetadata.from_yaml_file(bento_yaml)

        return cls(bento_metadata.tag, bento_fs)

    @cached_property
    def metadata(self):
        with self.fs.open(BENTO_YAML_FILENAME, "r") as bento_yaml:
            return BentoMetadata.from_yaml_file(bento_yaml)

    def creation_time(self) -> datetime.datetime:
        return self.metadata.creation_time

    def save(self, bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store]):
        if not self.validate():
            logger.warning(f"Failed to create Bento for {self.tag}, not saving.")
            raise BentoMLException("Failed to save Bento because it was invalid")

        with bento_store.register(self.tag) as bento_path:
            # TODO: handle non-OSFS store types
            os.rmdir(bento_path)
            os.rename(self.fs.getsyspath("/"), bento_path)
            self.fs.close()

    def export(self, path: str):
        mirror(
            self.fs, fs.open_fs(path, create=True, writeable=True), copy_if_newer=False
        )

    def push(self):
        pass

    def validate(self):
        return self.fs.isfile(BENTO_YAML_FILENAME)


class BentoStore(Store[Bento]):
    def __init__(self, base_path: PathType):
        super().__init__(base_path, Bento)


class BentoMetadata(UserDict):
    def dump(self, stream: t.TextIO):
        return yaml.dump(self.data, stream, sort_keys=False)

    @classmethod
    def from_build_args(
        cls, tag: Tag, svc: "Service", labels: t.Dict[str, t.Any], models: t.List[str]
    ):
        bento_metadata = cls()
        bento_metadata["service"] = svc._import_str
        bento_metadata["name"] = tag.name
        bento_metadata["version"] = tag.version
        bento_metadata["bentoml_version"] = __version__
        bento_metadata["created_at"] = datetime.datetime.now().isoformat()
        bento_metadata["labels"] = labels  # TODO: validate user provided labels
        apis = {}
        for api in svc._apis.values():
            apis[api.name] = dict(
                route=api.route,
                doc=api.doc,
                input=api.input.__class__.__name__,
                output=api.output.__class__.__name__,
            )
        bento_metadata["apis"] = apis
        bento_metadata["models"] = models  # TODO: populate with model & framework info

        bento_metadata.validate()
        return bento_metadata

    @classmethod
    def from_yaml_file(cls, stream: t.TextIO):
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

        bento_metadata = cls()
        bento_metadata.update(yaml_content)
        bento_metadata.validate()
        return bento_metadata

    def validate(self):
        # Validate bento.yml file schema, content, bentoml version, etc
        ...

    @property
    def creation_time(self):
        return datetime.datetime.fromisoformat(self["created_at"])

    @property
    def tag(self):
        return Tag(self["name"], self["version"])
