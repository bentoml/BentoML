"""
User facing python APIs for managing local bentos and build new bentos
"""
import os
import typing as t
import logging
import subprocess
from typing import TYPE_CHECKING

import fs
from simple_di import inject
from simple_di import Provide

from .exceptions import InvalidArgument
from ._internal.bento import Bento
from ._internal.types import Tag
from ._internal.utils import resolve_user_filepath
from ._internal.bento.build_config import BentoBuildConfig
from ._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ._internal.bento import BentoStore
    from ._internal.models import ModelStore

logger = logging.getLogger(__name__)


@inject
def list(  # pylint: disable=redefined-builtin
    tag: t.Optional[t.Union[Tag, str]] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> "t.List[Bento]":
    return _bento_store.list(tag)


@inject
def get(
    tag: t.Union[Tag, str],
    *,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> Bento:
    return _bento_store.get(tag)


@inject
def delete(
    tag: t.Union[Tag, str],
    *,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
):
    _bento_store.delete(tag)


@inject
def import_bento(
    path: str,
    *,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> Bento:
    return Bento.from_fs(fs.open_fs(path))


@inject
def export_bento(
    tag: t.Union[Tag, str],
    path: str,
    *,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
):
    bento = get(tag, _bento_store=_bento_store)
    bento.export(path)


@inject
def push(
    tag: t.Union[Tag, str],
    *,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
):
    raise NotImplementedError


@inject
def pull(
    tag: t.Union[Tag, str],
    *,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
):
    raise NotImplementedError


@inject
def build(
    service: str,
    *,
    labels: t.Optional[t.Dict[str, str]] = None,
    description: t.Optional[str] = None,
    include: t.Optional[t.List[str]] = None,
    exclude: t.Optional[t.List[str]] = None,
    additional_models: t.Optional[t.List[t.Union[str, Tag]]] = None,
    docker: t.Optional[t.Dict[str, t.Any]] = None,
    python: t.Optional[t.Dict[str, t.Any]] = None,
    conda: t.Optional[t.Dict[str, t.Any]] = None,
    version: t.Optional[str] = None,
    build_ctx: t.Optional[str] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "Bento":
    """
    User-facing API for building a Bento, the available build options are symmetrical to
    the content of a valid bentofile.yaml file, for building Bento from CLI.

    This API will not respect bentofile.yaml file in current environment, build options
    can only be provided via function call parameters.

    Args:
        service: import str for finding the bentoml.Service instance build target
        labels: optional immutable labels for carrying contextual info
        description: optional description string in markdown format
        include: list of file paths and patterns specifying files to include in Bento,
            default is all files under build_ctx, beside the ones excluded from the
            exclude parameter or a .bentoignore file for a given directory
        exclude: list of file paths and patterns to exclude from the final Bento archive
        additional_models: list of model tags to pack in Bento, in addition to the
            models that are required by service's runners. These models must be
            found in the given _model_store
        docker: dictionary for configuring Bento's containerization process, see details
            in `._internal.bento.build_config.DockerOptions`
        python: dictionary for configuring Bento's python dependencies, see details in
            `._internal.bento.build_config.PythonOptions`
        conda: dictionary for configuring Bento's conda dependencies, see details in
            `._internal.bento.build_config.CondaOptions`
        version: Override the default auto generated version str
        build_ctx: Build context directory, when used as
        _bento_store: save Bento created to this BentoStore
        _model_store: pull Models required from this ModelStore

    Returns:
        Bento: a Bento instance representing the materialized Bento saved in BentoStore

    """  # noqa: LN001
    build_config = BentoBuildConfig(
        service=service,
        description=description,
        labels=labels,
        include=include,
        exclude=exclude,
        additional_models=additional_models,
        docker=docker,
        python=python,
        conda=conda,
    )

    bento = Bento.create(
        build_config=build_config,
        version=version,
        build_ctx=build_ctx,
        model_store=_model_store,
    ).save(_bento_store)
    logger.info("Bento build success, %s created", bento)
    return bento


@inject
def build_bentofile(
    bentofile: str = "bentofile.yaml",
    *,
    version: t.Optional[str] = None,
    build_ctx: t.Optional[str] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "Bento":
    """
    Build a Bento base on options specified in a bentofile.yaml file.

    By default, this function will look for a `bentofile.yaml` file in current working
    directory.

    Args:
        bentofile: The file path to build config yaml file
        version: Override the default auto generated version str
        build_ctx: Build context directory, when used as
        _bento_store: save Bento created to this BentoStore
        _model_store: pull Models required from this ModelStore
    """
    try:
        bentofile = resolve_user_filepath(bentofile, build_ctx)
    except FileNotFoundError:
        raise InvalidArgument(f'bentofile "{bentofile}" not found')

    with open(bentofile, "r", encoding="utf-8") as f:
        build_config = BentoBuildConfig.from_yaml(f)

    bento = Bento.create(
        build_config=build_config,
        version=version,
        build_ctx=build_ctx,
        model_store=_model_store,
    ).save(_bento_store)
    logger.info("Bento build success, %s created", bento)
    return bento


@inject
def containerize(
    tag: t.Union[Tag, str],
    docker_image_tag: t.Optional[str] = None,
    *,
    build_args: t.Optional[t.Dict[str, str]] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    no_cache: bool = False,
    platform: t.Optional[str] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
):
    bento = _bento_store.get(tag)
    if docker_image_tag is None:
        docker_image_tag = str(bento.tag)

    docker_build_cmd = ["docker", "build", "."]
    dockerfile_path = os.path.join("env", "docker", "Dockerfile")
    docker_build_cmd += ["-f", dockerfile_path]
    docker_build_cmd += ["-t", docker_image_tag]

    if no_cache:
        docker_build_cmd.append("--no-cache")

    if build_args:
        for key, value in build_args.items():
            docker_build_cmd += ["--build-arg", f"{key}={value}"]

    if labels:
        for key, value in labels.items():
            docker_build_cmd += ["--label", f"{key}={value}"]

    if platform:
        docker_build_cmd += ["--platform", platform]

    env = os.environ.copy()
    env["DOCKER_SCAN_SUGGEST"] = "false"
    logger.info(f"Building docker image for {bento}...")
    subprocess.check_output(docker_build_cmd, cwd=bento.path, env=env)
    logger.info(f'Successfully built docker image "{docker_image_tag}"')


__all__ = [
    "list",
    "get",
    "delete",
    "import_bento",
    "export_bento",
    "push",
    "pull",
    "build",
    "build_bentofile",
    "containerize",
]
