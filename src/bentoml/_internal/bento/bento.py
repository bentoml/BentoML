from __future__ import annotations

import json
import logging
import os
import typing as t
from datetime import datetime
from datetime import timezone
from typing import TYPE_CHECKING

import attr
import fs
import fs.errors
import fs.mirror
import fs.osfs
import fs.walk
import yaml
from cattr.gen import make_dict_structure_fn
from cattr.gen import make_dict_unstructure_fn
from cattr.gen import override
from fs.copy import copy_file
from fs.tempfs import TempFS
from simple_di import Provide
from simple_di import inject

from ...exceptions import BentoMLException
from ...exceptions import InvalidArgument
from ...exceptions import NotFound
from ..configuration import BENTOML_VERSION
from ..configuration.containers import BentoMLContainer
from ..models import ModelStore
from ..models import copy_model
from ..runner import Runner
from ..store import Store
from ..store import StoreItem
from ..tag import Tag
from ..tag import to_snake_case
from ..types import PathType
from ..utils import normalize_labels_value
from ..utils.cattr import bentoml_cattr
from ..utils.filesystem import copy_file_to_fs_folder
from ..utils.uri import encode_path_for_uri
from .build_config import BentoBuildConfig
from .build_config import BentoEnvSchema
from .build_config import BentoPathSpec
from .build_config import CondaOptions
from .build_config import DockerOptions
from .build_config import PythonOptions

if TYPE_CHECKING:
    from fs.base import FS

    from _bentoml_sdk import Service as NewService
    from _bentoml_sdk.models import Model
    from _bentoml_sdk.service import ServiceConfig
    from _bentoml_sdk.service.dependency import Dependency

    from ..cloud.schemas.modelschemas import BentoManifestSchema
    from ..models import Model as StoredModel
    from ..service import Service
    from ..service.inference_api import InferenceAPI
else:
    ServiceConfig = t.Dict[str, t.Any]

logger = logging.getLogger(__name__)

BENTO_YAML_FILENAME = "bento.yaml"
BENTO_PROJECT_DIR_NAME = "src"
BENTO_README_FILENAME = "README.md"
DEFAULT_BENTO_BUILD_FILES = ("bentofile.yaml", "pyproject.toml")

API_INFO_MD = "| POST [`/{api}`](#{link}) | {input} | {output} |"

INFERENCE_TABLE_MD = """\
| InferenceAPI | Input | Output |
| ------------ | ----- | ------ |
{content}
"""


def create_inference_api_table(svc: Service) -> str:
    from ..service.openapi import APP_TAG

    contents = [
        API_INFO_MD.format(
            api=api.name,
            link=f"operations-{APP_TAG.name.replace(' ', '_')}-{svc.name}__{api.name}",  # follows operationId from OpenAPI
            input=api.input.__class__.__name__,
            output=api.output.__class__.__name__,
        )
        for api in svc.apis.values()
    ]
    return INFERENCE_TABLE_MD.format(content="\n".join(contents))


def get_default_svc_readme(
    svc: Service | NewService[t.Any], svc_version: str | None = None
) -> str:
    from ..service import Service

    if svc.bento:
        bentoml_version = svc.bento.info.bentoml_version
    else:
        bentoml_version = BENTOML_VERSION

    is_legacy = isinstance(svc, Service)

    if not svc_version:
        if svc.bento and svc.bento.tag.version:
            svc_version = svc.bento.tag.version
        else:
            svc_version = "dev"

    doc = f"""\
# {svc.name}:{svc_version}

[![pypi_status](https://img.shields.io/badge/BentoML-{bentoml_version}-informational)](https://pypi.org/project/BentoML)
[![documentation_status](https://readthedocs.org/projects/bentoml/badge/?version=latest)](https://docs.bentoml.com/)
[![join_slack](https://badgen.net/badge/Join/BentoML%20Slack/cyan?icon=slack)](https://l.bentoml.com/join-slack-swagger)
[![BentoML GitHub Repo](https://img.shields.io/github/stars/bentoml/bentoml?style=social)](https://github.com/bentoml/BentoML)
[![Twitter Follow](https://img.shields.io/twitter/follow/bentomlai?label=Follow%20BentoML&style=social)](https://twitter.com/bentomlai)

This is a Machine Learning Service created with BentoML."""

    if is_legacy and svc.apis:
        doc += f"\n{create_inference_api_table(svc)}\n\n"

    doc += """

## Help

* [📖 Documentation](https://docs.bentoml.com/en/latest/): Learn how to use BentoML.
* [💬 Community](https://l.bentoml.com/join-slack-swagger): Join the BentoML Slack community.
* [🐛 GitHub Issues](https://github.com/bentoml/BentoML/issues): Report bugs and feature requests.
* Tip: you can also [customize this README](https://docs.bentoml.com/en/latest/reference/bentoml/bento-build-options.html#description).
"""
    # TODO: add links to documentation that may help with API client development
    return doc


@attr.define(repr=False, auto_attribs=False)
class Bento(StoreItem):
    _tag: Tag = attr.field()
    __fs: FS = attr.field()

    _info: BaseBentoInfo

    _model_store: ModelStore | None = None
    _doc: str | None = None

    @staticmethod
    def _export_ext() -> str:
        return "bento"

    @__fs.validator  # type:ignore # attrs validators not supported by pyright
    def check_fs(self, _attr: t.Any, new_fs: FS):
        try:
            models = new_fs.opendir("models")
        except fs.errors.ResourceNotFound:
            return
        else:
            self._model_store = ModelStore(models)

    def __init__(self, tag: Tag, bento_fs: "FS", info: "BaseBentoInfo"):
        self._tag = tag
        self.__fs = bento_fs
        self.check_fs(None, bento_fs)
        self._info = info

    @property
    def tag(self) -> Tag:
        return self._tag

    @property
    def _fs(self) -> FS:
        return self.__fs

    @property
    def info(self) -> BaseBentoInfo:
        return self._info

    @property
    def entry_service(self) -> str:
        return self.info.entry_service

    def get_manifest(self, dev: bool = False) -> BentoManifestSchema:
        from ..cloud.schemas.modelschemas import BentoManifestSchema
        from ..cloud.schemas.modelschemas import BentoRunnerResourceSchema
        from ..cloud.schemas.modelschemas import BentoRunnerSchema

        info = self.info
        models = [str(m.tag) for m in info.all_models]
        if isinstance(info, BentoInfo):
            runners = [
                BentoRunnerSchema(
                    name=r.name,
                    runnable_type=r.runnable_type,
                    models=r.models,
                    resource_config=(
                        BentoRunnerResourceSchema(
                            cpu=r.resource_config.get("cpu"),
                            nvidia_gpu=r.resource_config.get("nvidia.com/gpu"),
                            custom_resources=r.resource_config.get("custom_resources"),
                        )
                        if r.resource_config
                        else None
                    ),
                )
                for r in info.runners
            ]
            image: ImageInfo | None = None
        else:
            assert isinstance(info, BentoInfoV2)
            image = info.image
            runners = []
        return BentoManifestSchema(
            name=info.name,
            entry_service=info.entry_service,
            service=info.service,
            bentoml_version=info.bentoml_version,
            apis={},
            models=models,
            runners=runners,
            size_bytes=self.total_size(),
            services=info.services,
            envs=info.envs,
            schema=info.schema,
            version=info.version,
            dev=dev,
            image=image,
            spec=info.spec,
        )

    @classmethod
    @inject
    def create(
        cls,
        build_config: BentoBuildConfig,
        version: t.Optional[str] = None,
        build_ctx: t.Optional[str] = None,
        platform: t.Optional[str] = None,
        bare: bool = False,
        reload: bool = False,
        enabled_features: list[str] = Provide[BentoMLContainer.enabled_features],
    ) -> Bento:
        from _bentoml_sdk.images import Image
        from _bentoml_sdk.images import populate_image_from_build_config
        from _bentoml_sdk.models import BentoModel

        from ..service import Service
        from ..service.loader import load

        build_ctx = (
            os.getcwd()
            if build_ctx is None
            else os.path.realpath(os.path.expanduser(build_ctx))
        )
        if not os.path.isdir(build_ctx):
            raise InvalidArgument(
                f"Bento build context {build_ctx} does not exist or is not a directory."
            )

        BentoMLContainer.model_aliases.set(build_config.model_aliases)
        # This also verifies that svc can be imported correctly
        svc = load(build_config.service, working_dir=build_ctx, reload=reload)
        if not build_config.service:
            object.__setattr__(build_config, "service", get_service_import_str(svc))
        is_legacy = isinstance(svc, Service)
        # Apply default build options
        image: Image | None = None
        disable_image = "no_image" in enabled_features or is_legacy

        if isinstance(svc, Service):
            # for < 1.2
            bento_name = (
                build_config.name if build_config.name is not None else svc.name
            )
        else:
            # for >= 1.2
            svc.inject_config()
            bento_name = (
                build_config.name
                if build_config.name is not None
                else to_snake_case(svc.name)
            )
            build_config.envs.extend(svc.envs)
            build_config.labels.update(svc.labels)
            if svc.image is not None:
                image = svc.image
        if not disable_image:
            image = populate_image_from_build_config(image, build_config, build_ctx)
        build_config = build_config.with_defaults()
        tag = Tag(bento_name, version)
        if version is None:
            tag = tag.make_new_version()

        logger.debug(
            'Building BentoML service "%s" from build context "%s".', tag, build_ctx
        )
        bento_fs = TempFS(
            identifier=f"bentoml_bento_{bento_name}",
            temp_dir=BentoMLContainer.tmp_bento_store_dir.get(),
        )
        models: list[BentoModelInfo] = []

        def append_model(model: BentoModelInfo) -> None:
            if model not in models:
                models.append(model)

        if build_config.models:
            for model_spec in build_config.models:
                model = BentoModel(model_spec.tag)
                append_model(model.to_info(model_spec.alias))
        elif is_legacy:
            # XXX: legacy way to get models from service
            # Add all models required by the service
            for model in svc.models:
                append_model(BentoModel(model.tag).to_info())
            # Add all models required by service runners
            for runner in svc.runners:
                for model in runner.models:
                    append_model(BentoModel(model.tag).to_info())

        if not bare:
            ctx_fs = fs.open_fs(encode_path_for_uri(build_ctx))

            # create ignore specs
            specs = BentoPathSpec(build_config.include, build_config.exclude, build_ctx)

            # Copy all files base on include and exclude, into `src` directory
            relpaths = [s for s in build_config.include if s.startswith("../")]
            if len(relpaths) != 0:
                raise InvalidArgument(
                    "Paths outside of the build context directory cannot be included; use a symlink or copy those files into the working directory manually."
                )
            bento_fs.makedir(BENTO_PROJECT_DIR_NAME)
            target_fs = bento_fs.opendir(BENTO_PROJECT_DIR_NAME)
            with target_fs.open("bentofile.yaml", "w") as bentofile_yaml:
                build_config.to_yaml(bentofile_yaml)

            for dir_path, _, files in ctx_fs.walk():
                for f in files:
                    path = fs.path.combine(dir_path, f.name).lstrip("/")
                    if specs.includes(path):
                        if ctx_fs.getsize(path) > 10 * 1024 * 1024:
                            logger.warning("File size is larger than 10MiB: %s", path)
                        target_fs.makedirs(dir_path, recreate=True)
                        copy_file(ctx_fs, path, target_fs, path)
            if image is None:
                # NOTE: we need to generate both Python and Conda
                # first to make sure we can generate the Dockerfile correctly.
                build_config.python.write_to_bento(
                    bento_fs, build_ctx, platform_=platform
                )
                build_config.conda.write_to_bento(bento_fs, build_ctx)
                build_config.docker.write_to_bento(
                    bento_fs, build_ctx, build_config.conda
                )

            # Create `readme.md` file
            if (
                build_config.description is not None
                and build_config.description.startswith("file:")
            ):
                file_name = build_config.description[5:].strip()
                if not ctx_fs.exists(file_name):
                    raise InvalidArgument(f"File {file_name} does not exist.")
                copy_file_to_fs_folder(
                    ctx_fs.getsyspath(file_name),
                    bento_fs,
                    dst_filename=BENTO_README_FILENAME,
                )
            elif build_config.description is None and ctx_fs.exists(
                BENTO_README_FILENAME
            ):
                copy_file_to_fs_folder(
                    ctx_fs.getsyspath(BENTO_README_FILENAME),
                    bento_fs,
                    dst_filename=BENTO_README_FILENAME,
                )
            else:
                with bento_fs.open(BENTO_README_FILENAME, "w") as f:
                    if build_config.description is None:
                        f.write(get_default_svc_readme(svc, version))
                    else:
                        f.write(build_config.description)

            # Create 'apis/openapi.yaml' file
            bento_fs.makedir("apis")
            with bento_fs.open(fs.path.combine("apis", "openapi.yaml"), "w") as f:
                yaml.dump(svc.openapi_spec, f)
            if not is_legacy:
                bentoschema = svc.schema()
                with bento_fs.open(fs.path.combine("apis", "schema.json"), "w") as f:
                    json.dump(bentoschema, f, indent=2)
                openai_endpoint = None
                _suffix = "/chat/completions"
                _len_suffix = len(_suffix)

                try:
                    first_match = next(
                        p
                        for p in bentoschema["paths"]
                        if isinstance(p, str) and p.endswith(_suffix)
                    )
                    openai_endpoint = first_match[:-_len_suffix]
                except StopIteration:
                    pass
                except KeyError:
                    # "paths" should never be missing when creating new bentos.
                    # but at least warn the user.
                    logger.warning(
                        "'paths' not found in generated schemas. Please report this issue upstream at https://github.com/bentoml/BentoML/issues"
                    )
                    pass

                if openai_endpoint is not None:
                    build_config.labels["openai_endpoint"] = openai_endpoint

        if image is None:
            bento_info = BentoInfo(
                tag=tag,
                service=svc,  # type: ignore # attrs converters do not typecheck
                entry_service=svc.name,
                labels=build_config.labels,
                models=models,
                runners=(
                    [BentoRunnerInfo.from_runner(r) for r in svc.runners]  # type: ignore # attrs converters do not typecheck
                    if is_legacy
                    else []
                ),
                apis=(
                    [BentoApiInfo.from_inference_api(api) for api in svc.apis.values()]
                    if is_legacy
                    else []
                ),
                services=(
                    [
                        BentoServiceInfo.from_service(s)
                        for s in svc.all_services().values()
                    ]
                    if not is_legacy
                    else []
                ),
                docker=build_config.docker,
                python=build_config.python,
                conda=build_config.conda,
                envs=build_config.envs,
                schema=svc.schema() if not is_legacy else {},
            )
        else:
            bento_info = BentoInfoV2(
                tag=tag,
                service=svc,  # type: ignore # attrs converters do not typecheck
                entry_service=svc.name,
                labels=build_config.labels,
                models=models,
                services=(
                    [
                        BentoServiceInfo.from_service(s)
                        for s in svc.all_services().values()
                    ]
                    if not is_legacy
                    else []
                ),
                envs=build_config.envs,
                schema=svc.schema() if not is_legacy else {},
                image=image.freeze(bento_fs, build_config.envs, platform),
            )

        res = Bento(tag, bento_fs, bento_info)
        if bare:
            return res
        # Create bento.yaml
        res.flush_info()
        try:
            res.validate()
        except BentoMLException as e:
            raise BentoMLException(f"Failed to create {res!s}: {e}") from None

        return res

    @classmethod
    def from_fs(cls, item_fs: FS) -> Bento:
        try:
            with item_fs.open(BENTO_YAML_FILENAME, "r", encoding="utf-8") as bento_yaml:
                info = BaseBentoInfo.from_yaml_file(bento_yaml)
        except fs.errors.ResourceNotFound:
            raise BentoMLException(
                f"Failed to load bento because it does not contain a '{BENTO_YAML_FILENAME}'"
            )

        res = cls(info.tag, item_fs, info)
        try:
            res.validate()
        except BentoMLException as e:
            raise BentoMLException(f"Failed to load bento: {e}") from None

        return res

    @classmethod
    def from_path(cls, item_path: str) -> Bento:
        item_path = os.path.expanduser(item_path)
        item_fs = fs.open_fs(encode_path_for_uri(item_path))
        return cls.from_fs(item_fs)

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
        from _bentoml_sdk.models import BentoModel

        # copy the models to fs
        models_dir = self._fs.makedir("models", recreate=True)
        for model in self.info.all_models:
            if model.registry != "bentoml":
                continue
            BentoModel(model.tag).resolve(models_dir)
        try:
            return super().export(
                path,
                output_format,
                protocol=protocol,
                user=user,
                passwd=passwd,
                params=params,
                subpath=subpath,
            )
        finally:
            self._fs.removetree("models")

    @inject
    def total_size(
        self, model_store: ModelStore = Provide[BentoMLContainer.model_store]
    ) -> int:
        total_size = self.file_size
        local_model_store = self._model_store
        for model in self.info.all_models:
            if model.registry == "huggingface":
                continue
            if local_model_store is not None:
                try:
                    local_model_store.get(model.tag)
                    continue
                except NotFound:
                    pass
            try:
                global_model = model_store.get(model.tag)
            except NotFound:
                logger.warning(
                    "Bento: %s: Missing model %s",
                    self.tag,
                    model.tag,
                )
                continue
            total_size += global_model.file_size
        return total_size

    def flush_info(self):
        with self._fs.open(BENTO_YAML_FILENAME, "w") as bento_yaml:
            self.info.dump(bento_yaml)

    @property
    def doc(self) -> str:
        if self._doc is not None:
            return self._doc
        if not self._fs.isfile(BENTO_README_FILENAME):
            return ""
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
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ) -> "Bento":
        try:
            self.validate()
        except BentoMLException as e:
            raise BentoMLException(f"Failed to save {self!s}: {e}") from None

        with bento_store.register(self.tag) as bento_path:
            out_fs = fs.open_fs(
                encode_path_for_uri(bento_path), create=True, writeable=True
            )
            if self._model_store is not None:
                # Move models to the global model store, if any
                for model in self._model_store.list():
                    copy_model(
                        model.tag,
                        src_model_store=self._model_store,
                        target_model_store=model_store,
                    )
                self._model_store = None
            fs.mirror.mirror(self._fs, out_fs, copy_if_newer=False)
            try:
                out_fs.removetree("models")
            except fs.errors.ResourceNotFound:
                pass
            self._fs.close()
            self.__fs = out_fs

        return self

    def validate(self):
        if not self._fs.isfile(BENTO_YAML_FILENAME):
            raise BentoMLException(
                f"{self!s} does not contain a {BENTO_YAML_FILENAME}."
            )

    def __str__(self):
        return f'Bento(tag="{self.tag}")'


class BentoStore(Store[Bento]):
    def __init__(self, base_path: t.Union[PathType, "FS"]):
        super().__init__(base_path, Bento)


@attr.frozen
class BentoRunnerInfo:
    name: str
    runnable_type: str
    embedded: bool = attr.field(default=False)
    models: t.List[str] = attr.field(factory=list)
    resource_config: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)

    @classmethod
    def from_runner(cls, r: Runner) -> BentoRunnerInfo:
        return cls(
            name=r.name,
            runnable_type=r.runnable_class.__name__,
            embedded=r.embedded,
            models=[str(model.tag) for model in r.models],
            resource_config=r.resource_config,
        )


@attr.frozen
class BentoApiInfo:
    name: str
    input_type: str
    output_type: str

    @classmethod
    def from_inference_api(cls, api: InferenceAPI[t.Any]) -> BentoApiInfo:
        return cls(
            name=api.name,
            input_type=api.input.__class__.__name__,
            output_type=api.output.__class__.__name__,
        )


@attr.frozen
class BentoModelInfo:
    tag: Tag = attr.field(converter=Tag.from_taglike)
    module: t.Optional[str] = attr.field(default=None, eq=False)
    creation_time: datetime = attr.field(
        factory=lambda: datetime.now(timezone.utc), eq=False
    )
    alias: t.Optional[str] = attr.field(default=None, eq=False)
    registry: str = attr.field(default="bentoml", eq=False)
    metadata: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None, eq=False)

    @classmethod
    def from_bento_model(
        cls, bento_model: StoredModel, alias: str | None = None
    ) -> BentoModelInfo:
        return cls(
            tag=bento_model.tag,
            module=bento_model.info.module,
            creation_time=bento_model.info.creation_time,
            alias=alias,
        )

    def to_model(self) -> Model[t.Any]:
        from bentoml.models import BentoModel
        from bentoml.models import HuggingFaceModel

        if self.registry == "bentoml":
            return BentoModel.from_info(self)
        elif self.registry == "huggingface":
            return HuggingFaceModel.from_info(self)
        else:
            raise ValueError(f"Unknown model registry: {self.registry}")


@attr.frozen
class BentoDependencyInfo:
    service: t.Optional[str] = None
    deployment: t.Optional[str] = None
    cluster: t.Optional[str] = None
    url: t.Optional[str] = None

    @classmethod
    def from_dependency(cls, d: Dependency[t.Any]) -> BentoDependencyInfo:
        return cls(
            service=d.on.name if d.on is not None else None,
            deployment=d.deployment,
            cluster=d.cluster,
            url=d.url,
        )


@attr.frozen
class BentoServiceInfo:
    name: str
    service: str
    models: t.List[BentoModelInfo] = attr.field(factory=list, eq=False)
    dependencies: t.List[BentoDependencyInfo] = attr.field(factory=list, eq=False)
    config: ServiceConfig = attr.field(factory=dict, eq=False)

    @classmethod
    def from_service(cls, svc: NewService[t.Any]) -> BentoServiceInfo:
        return cls(
            name=svc.name,
            service="",
            models=[m.to_info() for m in svc.models],
            dependencies=[
                BentoDependencyInfo.from_dependency(d)
                for d in svc.dependencies.values()
            ],
            config=svc.config,
        )


def get_service_import_str(svc: Service | NewService[t.Any] | str) -> str:
    from ..service import Service

    if isinstance(svc, Service):
        return svc.get_service_import_origin()[0]
    if isinstance(svc, str):
        return svc
    return svc.import_string


@attr.frozen(repr=False)
class BaseBentoInfo:
    # for backward compatibility in case new fields are added to BentoInfo.
    __forbid_extra_keys__ = False
    # omit field in yaml file if it is not provided by the user.
    __omit_if_default__ = True

    tag: Tag
    service: str = attr.field(converter=get_service_import_str)
    name: str = attr.field(init=False)
    version: str = attr.field(init=False)
    # using factory explicitly instead of default because omit_if_default is enabled for BentoInfo
    bentoml_version: str = attr.field(factory=lambda: BENTOML_VERSION)
    creation_time: datetime = attr.field(factory=lambda: datetime.now(timezone.utc))

    labels: t.Dict[str, t.Any] = attr.field(
        factory=dict, converter=normalize_labels_value
    )
    models: t.List[BentoModelInfo] = attr.field(factory=list)
    # for BentoML 1.2+ SDK
    entry_service: str = attr.field(factory=str)
    services: t.List[BentoServiceInfo] = attr.field(factory=list)
    envs: t.List[BentoEnvSchema] = attr.field(factory=list)
    schema: t.Dict[str, t.Any] = attr.field(factory=dict)
    args: t.Dict[str, t.Any] = attr.field(factory=BentoMLContainer.bento_arguments.get)

    @property
    def all_models(self) -> t.List[BentoModelInfo]:
        model_map = {model.tag: model for model in self.models}
        for service in self.services:
            for model in service.models:
                model_map[model.tag] = model
        return list(model_map.values())

    def __attrs_post_init__(self):
        # Direct set is not available when frozen=True
        object.__setattr__(self, "name", self.tag.name)
        object.__setattr__(self, "version", self.tag.version)

        try:
            self.validate()
        except BentoMLException as e:
            raise BentoMLException(f"Failed to initialize {self!s}: {e}") from None

    def to_dict(self) -> t.Dict[str, t.Any]:
        return bentoml_cattr.unstructure(self)

    def dump(self, stream: t.IO[t.Any]) -> None:
        # _models is an alias for models, replace it with models
        return yaml.safe_dump(self.to_dict(), stream, sort_keys=False)

    @classmethod
    def from_yaml_file(cls, stream: t.IO[t.Any]) -> BaseBentoInfo:
        try:
            yaml_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error("Error while parsing YAML file: %s", exc)
            raise

        assert yaml_content is not None

        yaml_content["tag"] = Tag(yaml_content["name"], yaml_content["version"])
        del yaml_content["name"]
        del yaml_content["version"]

        # For backwards compatibility for bentos created prior to version 1.0.0rc1
        if "runners" in yaml_content:
            runners = yaml_content["runners"]
            for r in runners:
                if "runner_type" in r:  # BentoRunnerInfo prior to 1.0.0rc1 release
                    r["runnable_type"] = r["runner_type"]
                    del r["runner_type"]
                    if "model_runner_module" in r:
                        del r["model_runner_module"]

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
        if yaml_content.pop("spec", 1) == 2:
            klass = BentoInfoV2
        else:
            klass = BentoInfo
        try:
            return bentoml_cattr.structure(yaml_content, klass)
        except KeyError as e:
            raise BentoMLException(f"Missing field {e} in {BENTO_YAML_FILENAME}")

    def validate(self):
        # Validate bento.yml file schema, content, bentoml version, etc
        ...


@attr.frozen(repr=False)
class BentoInfo(BaseBentoInfo):
    spec: int = attr.field(default=1, init=False)
    runners: t.List[BentoRunnerInfo] = attr.field(factory=list)
    apis: t.List[BentoApiInfo] = attr.field(factory=list)
    docker: DockerOptions = attr.field(factory=lambda: DockerOptions().with_defaults())
    python: PythonOptions = attr.field(factory=lambda: PythonOptions().with_defaults())
    conda: CondaOptions = attr.field(factory=lambda: CondaOptions().with_defaults())


@attr.frozen(repr=False)
class ImageInfo:
    base_image: str = ""
    python_version: str = ""
    commands: t.List[str] = attr.field(factory=list)
    python_requirements: str = ""
    post_commands: t.List[str] = attr.field(factory=list)


@attr.frozen(repr=False)
class BentoInfoV2(BaseBentoInfo):
    spec: int = attr.field(default=2, init=False)
    image: ImageInfo = attr.field(factory=ImageInfo)


bentoml_cattr.register_unstructure_hook(
    BentoDependencyInfo,
    make_dict_unstructure_fn(
        BentoDependencyInfo,
        bentoml_cattr,
        service=override(omit_if_default=True),
        deployment=override(omit_if_default=True),
        cluster=override(omit_if_default=True),
        url=override(omit_if_default=True),
    ),
)


def _convert_bento_dependency_info(
    data: str | dict[str, t.Any], typ: type[BentoDependencyInfo]
) -> BentoDependencyInfo:
    if isinstance(data, str):
        return BentoDependencyInfo(service=data)
    return BentoDependencyInfo(**data)


bentoml_cattr.register_structure_hook(
    BentoDependencyInfo, _convert_bento_dependency_info
)

bentoml_cattr.register_unstructure_hook(
    BentoModelInfo,
    make_dict_unstructure_fn(
        BentoModelInfo,
        bentoml_cattr,
        alias=override(omit_if_default=True),
        registry=override(omit_if_default=True),
        metadata=override(omit_if_default=True),
    ),
)
bentoml_cattr.register_structure_hook_factory(
    lambda cls: issubclass(cls, BaseBentoInfo),
    lambda cls: make_dict_structure_fn(
        cls,
        bentoml_cattr,
        name=override(omit=True),
        version=override(omit=True),
    ),
)
bentoml_cattr.register_unstructure_hook_factory(
    lambda cls: issubclass(cls, BaseBentoInfo),
    # Ignore tag, tag is saved via the name and version field
    lambda cls: make_dict_unstructure_fn(cls, bentoml_cattr, tag=override(omit=True)),
)
