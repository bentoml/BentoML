import inspect
import itertools
import logging
import operator
import os
import shutil
import tarfile
import typing as t
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import attr
import yaml
from simple_di import Provide, inject

from bentoml import __version__ as BENTOML_VERSION

from ...exceptions import BentoMLException, InvalidArgument
from ..configuration.containers import BentoMLContainer
from ..types import PathType
from ..utils import generate_new_version_id, validate_or_create_dir
from . import MODEL_YAML_NAMESPACE, YAML_EXT

logger = logging.getLogger(__name__)

RESERVED_MODEL_FIELD = [
    "name",
    "path",
    "version",
    "module",
    "created_at",
    "labels",
    "context",
]
SUPPORTED_COMPRESSION_TYPE = [".gz"]
MODEL_TAR_EXTENSION = ".models.tar.gz"


def validate_name(name: str):
    if not name.isidentifier():
        raise InvalidArgument(
            f"Invalid model name: '{name}'. A valid identifier "
            "may only contain letters, numbers, underscores "
            "and not starting with a number."
        )


def _generate_model_tag(name: str):
    validate_name(name)
    version = generate_new_version_id()
    return f"{name}:{version}"


def _process_model_tag(tag: str) -> (str, str):
    try:
        _name, _version = tag.split(":")
        validate_name(_name)
        return _name, _version
    except ValueError:
        validate_name(tag)
        # when name is a model name without versioning
        return tag, "latest"


@attr.s
class StoreCtx(object):
    name = attr.ib(type=str)
    version = attr.ib(type=str)
    path = attr.ib(type=PathType)
    tag = attr.ib(type=str)
    labels = attr.ib(
        factory=list,
        type=t.List[str],
        kw_only=True,
        converter=attr.converters.default_if_none(list()),
    )
    options = attr.ib(
        factory=dict,
        type=t.Dict[str, t.Any],
        kw_only=True,
        converter=attr.converters.default_if_none(dict()),
    )
    metadata = attr.ib(
        factory=dict,
        type=t.Dict[str, t.Any],
        kw_only=True,
        converter=attr.converters.default_if_none(dict()),
    )


@attr.s
class ModelInfo(StoreCtx):
    context = attr.ib(
        factory=dict,
        type=t.Dict[str, t.Any],
        converter=attr.converters.default_if_none(dict()),
    )
    module = attr.ib(type=str, kw_only=True)
    created_at = attr.ib(type=str, kw_only=True)
    api_version = attr.ib(type=str, default="v1")
    bentoml_version = attr.ib(type=str, default=BENTOML_VERSION)


def dump_model_yaml(
    model_yaml: Path,
    ctx: "StoreCtx",
    *,
    framework_context: dict = None,
    module: str = __name__,
) -> None:
    info = ModelInfo(
        name=ctx.name,
        version=ctx.version,
        tag=ctx.tag,
        labels=ctx.labels,
        options=ctx.options,
        metadata=ctx.metadata,
        path=str(ctx.path),
        created_at=datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        context=framework_context,
        module=module,
    )
    with model_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(attr.asdict(info), f)


def load_model_yaml(path: PathType) -> "ModelInfo":
    with Path(path, f"{MODEL_YAML_NAMESPACE}{YAML_EXT}").open(
        "r", encoding="utf-8"
    ) as f:
        info = yaml.safe_load(f)
    return ModelInfo(**info)


class ModelStore:
    @inject
    def __init__(
        self,
        base_dir: PathType = Provide[BentoMLContainer.default_model_store_base_dir],
    ):
        self._base_dir = Path(base_dir)
        validate_or_create_dir(self._base_dir)

    def list(self, tag: t.Optional[str] = None) -> t.List[str]:
        """
        bentoml models list -> t.List[models name under BENTOML_HOME/models]
        bentoml models list my_nlp_models -> t.List[model_version]
        """
        if not tag:
            path = self._base_dir
            return sorted(
                list(
                    itertools.chain.from_iterable(
                        [
                            [f"{_d.name}:{ver}" for ver in self.list(_d.name)]
                            for _d in path.iterdir()
                        ]
                    )
                ),
                key=operator.itemgetter(0),
            )
        elif ":" not in tag:
            path = Path(self._base_dir, tag)
        else:
            name, version = _process_model_tag(tag)
            path = Path(self._base_dir, name, version)
            if version == "latest":
                path = path.resolve()
        return [_f.name for _f in path.iterdir() if not _f.is_symlink()]

    def _create_path(self, tag: str):
        name, version = _process_model_tag(tag)
        model_path = Path(self._base_dir, name, version)
        validate_or_create_dir(model_path)
        return model_path

    @contextmanager
    def register(
        self,
        name: str,
        *,
        module: str = "",
        labels: t.List[str] = None,
        options: t.Dict[str, t.Any] = None,
        metadata: t.Dict[str, t.Any] = None,
        framework_context: dict = None,
    ) -> "t.Iterator[StoreCtx]":
        """
        with bentoml.models.register(name, options, metadata, labels) as ctx:
            # ctx(model_path, version, metadata)
            model.save(ctx.model_path, metadata=ctx.metadata)
            ctx.metadata["params_a"] = value_a
        """
        tag = _generate_model_tag(name)
        _, version = tag.split(":")
        model_path = self._create_path(tag)
        model_yaml = Path(model_path, f"{MODEL_YAML_NAMESPACE}{YAML_EXT}")

        ctx = StoreCtx(
            name=name,
            version=version,
            tag=f"{name}:{version}",
            path=model_path,
            labels=labels,
            metadata=metadata,
            options=options,
        )
        try:
            yield ctx
        except Exception:  # noqa
            # save has failed
            logger.warning(f"Failed to save {tag}, deleting {model_path}...")
            shutil.rmtree(model_path)
        finally:
            latest_path = Path(self._base_dir, name, "latest")
            dump_model_yaml(
                model_yaml, ctx, framework_context=framework_context, module=module
            )
            if latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(model_path)

    def get(self, tag: str) -> "ModelInfo":
        """
        bentoml.pytorch.get("my_nlp_model")
        """
        name, version = _process_model_tag(tag)
        path = Path(self._base_dir, name, version)
        if not path.exists():
            raise FileNotFoundError(
                f"Model '{tag}' is not found under BentoML modelstore {self._base_dir}."
            )
        return load_model_yaml(path)

    def delete(self, tag: str, skip_confirm: bool = False):
        """
        bentoml models delete
        """
        model_name, version = _process_model_tag(tag)
        basepath = Path(self._base_dir, model_name)
        try:
            if ":" not in tag:
                basepath.rmdir()
            else:
                path = Path(basepath, version)
                path.rmdir()
                path.unlink(missing_ok=True)
        finally:
            indexed = sorted(basepath.iterdir(), key=os.path.getctime)
            latest_path = Path(basepath, "latest")
            if latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(indexed[-1])

    def push(self, tag: str):
        ...

    def pull(self, tag: str):
        ...

    def export(self, tag: str, exported_path: PathType) -> None:
        model_info = self.get(tag)
        fname = f"{model_info.name}_{model_info.version}{MODEL_TAR_EXTENSION}"
        exported_path = (
            os.path.expandvars(exported_path)
            if "$" in str(exported_path)
            else exported_path
        )
        with tarfile.open(os.path.join(exported_path, fname), mode="w:gz") as tfile:
            tfile.add(model_info.path, arcname="")

    def imports(self, path: PathType, override=False) -> str:
        _path_obj = Path(path)
        if _path_obj.suffix not in SUPPORTED_COMPRESSION_TYPE:
            raise BentoMLException(
                f"Compression type from {path} is not yet supported. "
                f"Currently supports: {SUPPORTED_COMPRESSION_TYPE}."
            )
        try:
            with tarfile.open(path, mode="r:gz") as tfile:
                with tfile.extractfile(
                    f"{MODEL_YAML_NAMESPACE}{YAML_EXT}"
                ) as model_yaml:
                    model_info = ModelInfo(**yaml.safe_load(model_yaml))
                    target = Path(self._base_dir, model_info.name, model_info.version)
                    validate_or_create_dir(target)
                    if not override and any(target.iterdir()):
                        raise FileExistsError
                tfile.extractall(path=str(target))
            return str(target)
        except FileExistsError:
            tag = f"{model_info.name}:{model_info.version}"
            model_info = self.get(tag)
            _LOAD_INST = """\
            import {module}
            model = {module}.load("{name}", **kwargs)
            """
            _LOAD_RUNNER_INST = """\
            import {module}
            runner = {module}.load_runner("{name}", **kwargs)
            """
            raise FileExistsError(
                f"Model `{tag}` have already been imported.\n"
                f"Import the model directly with `load`:\n\n"
                f"{inspect.cleandoc(_LOAD_INST.format(module=model_info.module, name=tag))}\n\n"
                f"Use runner directly with `load_runner`:\n\n"
                f"{inspect.cleandoc(_LOAD_RUNNER_INST.format(module=model_info.module, name=tag))}\n\n"
                f"If one wants to override, do\n\nbentoml.models.imports('{path}', override=True)"
            )
