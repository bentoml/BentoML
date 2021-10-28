import logging
import os
import typing as t

import attr
import fs
import pathspec
import yaml
from fs.base import FS
from fs.copy import copy_dir, copy_file
from simple_di import Provide, inject

from bentoml.exceptions import BentoMLException, InvalidArgument

from ..configuration.containers import BentoMLContainer
from ..types import PathType, Tag
from ..utils import generate_new_version_id
from ..utils.validation import validate_version_str

if t.TYPE_CHECKING:
    from bentoml._internal.service import Service

    from ..models.store import ModelStore
    from ..store import Store

logger = logging.getLogger(__name__)


@attr.define
class Bento:
    tag: Tag
    fs: FS

    @classmethod
    @inject
    def create(
        cls,
        svc: "Service",
        build_ctx: PathType,
        models: t.List[str] = [],
        version: t.Optional[str] = None,
        description: t.Optional[str] = None,
        include: t.List[str] = ["*"],
        exclude: t.List[str] = [],
        env: t.Optional[t.Dict[str, t.Any]] = None,
        labels: t.Optional[t.Dict[str, str]] = None,
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        if version is None:
            version = generate_new_version_id()

        validate_version_str(version)

        tag = Tag(svc.name, version)

        logger.debug(f"Building BentoML service {tag} from build context {build_ctx}")

        bento_fs = fs.open_fs(f"temp://bentoml_bento_{svc.name}")
        ctx_fs = fs.open_fs(build_ctx)

        # Copy required models from local modelstore, into
        # `models/{model_name}/{model_version}` directory
        for runner in svc._runners.values():
            models += runner.required_models

        # TODO: remove duplicates in the models list
        for model_tag in models:
            try:
                model_info = model_store.get(model_tag)
            except FileNotFoundError:
                raise BentoMLException(
                    f"Model {model_tag} not found in local model store"
                )

            model_name, model_version = model_tag.split(":")
            target_path = os.path.join("models", model_name, model_version)
            copy_dir(fs.open_fs(model_info.path), "/", bento_fs, target_path)

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
        # TODO

        # Create `readme.md` file
        description = svc.__doc__ if description is None else description
        with bento_fs.open("readme.md", "w") as f:
            f.write(description)

        # Create 'apis/openapi.yaml' file
        bento_fs.makedir("apis")
        with bento_fs.open(fs.path.combine("apis", "openapi.yaml"), "w") as f:
            yaml.dump(svc.openapi_doc(), f)

        # Create bento.yaml
        # TODO
        with bento_fs.open("bento.yaml", "w") as bento_yaml:  # noqa: F841
            pass

        return cls(tag, bento_fs)

    @classmethod
    def import_from_fs(cls, bento_fs: FS) -> "Bento":
        # TODO
        with bento_fs.open("bento.yaml", "r") as bento_yaml:  # noqa: F841
            pass
        return cls(Tag("TODO", None), bento_fs)

    def save(self, bento_store: "Store" = Provide[BentoMLContainer.bento_store]):
        if not self.validate():
            logger.warning(f"Failed to create Bento for {self.tag}, not saving.")
            raise BentoMLException("Failed to save Bento because it was invalid")

        with bento_store.register(self.tag) as bento_path:
            # TODO: handle non-OSFS store types
            os.rmdir(bento_path)
            os.rename(self.fs.getsyspath("/"), bento_path)
            self.fs.close()

    def validate(self):
        return self.fs.isfile("bento.yaml")
