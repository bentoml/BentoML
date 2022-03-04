import typing as t
import logging
from copy import deepcopy
from typing import TYPE_CHECKING
from pathlib import Path
from itertools import product

import attrs
from manager import DOCKERFILE_BUILD_HIERARCHY
from simple_di import inject
from simple_di import Provide
from manager._utils import walk
from manager._utils import as_posix
from manager._utils import raise_exception
from manager._utils import SUPPORTED_ARCHITECTURE_TYPE
from manager._container import RELEASE_PREFIX
from manager._container import ManagerContainer
from manager._container import get_general_context
from manager.exceptions import ManagerException
from manager._cuda_context import CUDACtx
from manager._cuda_context import CUDAVersion

if TYPE_CHECKING:
    from manager._types import StrList
    from manager._types import GenericDict
    from manager._types import GenericFunc
    from manager._types import ReleaseTagInfo

logger = logging.getLogger(__name__)


# file
EXTENSION = ".j2"
DOCKERFILE_NAME = "Dockerfile"
DOCKERFILE_TEMPLATE_SUFFIX = f"-{DOCKERFILE_NAME.lower()}{EXTENSION}"

# misc
ALL_DOCKER_SUPPORTED_ARCHITECTURE = [
    "amd64",
    "arm32v5",
    "arm32v6",
    "arm32v7",
    "arm64v8",
    "i386",
    "ppc64le",
    "s390x",
    "riscv64",
    "mips64le",
    "mips64",
]

__all__ = ["load_context"]


def _is_supported_by_docker(instance: t.Any, attribute: t.Any, value: t.List[str]):
    for v in value:
        if v not in ALL_DOCKER_SUPPORTED_ARCHITECTURE:
            raise ManagerException(f"{v} architecture is not supported by Docker.")
        if v not in SUPPORTED_ARCHITECTURE_TYPE:
            logger.debug(f"{v} is not yet supported by BentoML. Use with care!")


def _follow_build_hierarchy(instance: t.Any, attribute: t.Any, value: t.List[str]):
    for v in value:
        if v not in DOCKERFILE_BUILD_HIERARCHY:
            raise ManagerException(
                f"{v} is not a valid package release type. Valid: {DOCKERFILE_BUILD_HIERARCHY}"
            )


@attrs.define
class SharedCtx:
    suffixes: str
    distro_name: str
    bentoml_version: str
    python_version: str
    templates_dir: str = attrs.field(default=ManagerContainer.template_dir)
    generated_dir: str = attrs.field(default=ManagerContainer.generated_dir)
    docker_package: str = attrs.field(default=ManagerContainer.bento_server_name)
    packages: t.List[str] = attrs.field(
        default=attrs.Factory(list), validator=_follow_build_hierarchy
    )
    architectures: t.List[str] = attrs.field(
        default=attrs.Factory(list), validator=_is_supported_by_docker
    )


@raise_exception
def _process_envars(
    bentoml_version: str, python_version: str
) -> "GenericFunc[GenericDict]":
    args = {"BENTOML_VERSION": bentoml_version, "PYTHON_VERSION": python_version}

    def update_args_dict(
        updater: "t.Union[str, GenericDict, t.List[str]]",
    ) -> "GenericDict":
        # Args will have format ARG=foobar
        if isinstance(updater, str):
            arg, _, value = updater.partition("=")
            args[arg] = value
        elif isinstance(updater, list):
            for v in updater:
                update_args_dict(v)
        elif isinstance(updater, dict):
            for arg, value in updater.items():
                args[arg] = value
        else:
            logger.error(f"cannot add to args dict with unknown type {type(updater)}")
        return args

    return update_args_dict


@attrs.define
class BuildCtx:
    header: str
    base_image: str
    envars: t.Dict[str, str]
    cuda_ctx: CUDACtx = attrs.field(default=None)
    shared_ctx: SharedCtx = attrs.field(default=None)

    def __attrs_post_init__(self):
        self.envars = _process_envars(
            bentoml_version=self.shared_ctx.bentoml_version,
            python_version=self.shared_ctx.python_version,
        )(self.envars)


def _create_images_tags_from_context(
    release_type: t.List[str],
    python_version: str,
    suffixes: str,
    bentoml_version: str,
    docker_package: str = ManagerContainer.bento_server_name,
) -> "ReleaseTagInfo":
    # returns a list of strings following the build hierarchy
    TAG_FORMAT = "{release_type}-python{python_version}-{suffixes}"
    releases = {}
    for rt in release_type:
        if rt not in DOCKERFILE_BUILD_HIERARCHY:
            raise ManagerException(
                f"{rt} is not supported. "
                "If adding new releases type make sure to update DOCKERFILE_BUILD_HIERARCHY "
                "under `manager/__init__.py`."
            )
        if rt in ["runtime", "cudnn"]:
            tag = (
                docker_package
                + ":"
                + "-".join(
                    [
                        TAG_FORMAT.format(
                            release_type=bentoml_version,
                            python_version=python_version,
                            suffixes=suffixes,
                        ),
                        rt,
                    ]
                )
            )
        else:
            tag = (
                docker_package
                + ":"
                + TAG_FORMAT.format(
                    release_type=rt,
                    python_version=python_version,
                    suffixes=suffixes,
                )
            )

        if rt != "base":
            build_tag = TAG_FORMAT.format(
                release_type="base", python_version="$PYTHON_VERSION", suffixes=suffixes
            )
        else:
            build_tag = ""
        releases[rt] = {"image_tag": tag, "build_tag": build_tag}

    return releases


def _create_path_context(release_ctx: "ReleaseCtx") -> None:
    template_dir = ManagerContainer.docker_dir.joinpath(
        release_ctx.shared_ctx.templates_dir
    )

    release_tags = {}
    for release_type in release_ctx.shared_ctx.packages:
        target_path = Path(
            release_ctx.shared_ctx.docker_package,
            release_ctx.shared_ctx.distro_name,
            release_type,
        )
        rt = release_ctx.release_tags.pop(release_type)
        release_tags[rt["image_tag"]] = {
            "input_paths": [
                as_posix(f) for f in walk(template_dir) if release_type in f.name
            ],
            "output_path": as_posix(release_ctx.shared_ctx.generated_dir, target_path),
            "git_tree_path": as_posix(
                Path(release_ctx.shared_ctx.generated_dir).stem,
                target_path,
                DOCKERFILE_NAME,
            ),
            "build_tag": rt["build_tag"],
        }
    release_ctx.release_tags = release_tags


@attrs.define
class ReleaseCtx:
    shared_ctx: SharedCtx = attrs.field(default=None)
    release_tags: "ReleaseTagInfo" = attrs.field(default=attrs.Factory(dict))

    def __attrs_post_init__(self):
        self.release_tags = _create_images_tags_from_context(
            suffixes=self.shared_ctx.suffixes,
            release_type=self.shared_ctx.packages,
            python_version=self.shared_ctx.python_version,
            bentoml_version=self.shared_ctx.bentoml_version,
            docker_package=self.shared_ctx.docker_package,
        )
        _create_path_context(self)


@inject
def load_context(
    bentoml_version: str,
    generated_dir: str,
    python_version: t.Tuple[str],
    docker_package: str = ManagerContainer.bento_server_name,
    default_context: "GenericDict" = Provide[ManagerContainer.bento_server_general_ctx],
) -> "t.Tuple[t.Dict[str, BuildCtx], t.Dict[str, ReleaseCtx], t.Dict[str, StrList]]":
    if docker_package != ManagerContainer.bento_server_name:
        contexts = get_general_context(package=docker_package)
    else:
        contexts = default_context

    build_ctxs, release_ctxs, release_os = {}, {}, {}

    for pyver, (release_string, distros) in product(python_version, contexts.items()):
        cuda_version = release_string[len(RELEASE_PREFIX) :]
        major, minor, patch = cuda_version.split(".")
        los = []
        for oses, d in distros.items():
            los.append(oses)
            deps = deepcopy(d)
            cuda_ctx = CUDACtx(
                version=CUDAVersion(
                    major=major,
                    minor=minor,
                    patch=patch,
                    shortened=f"{major}.{minor}",
                    full=cuda_version,
                ),
                dependencies=deps.pop("dependencies"),
            )
            shared_ctx = SharedCtx(
                distro_name=oses,
                packages=deps.pop("packages"),
                suffixes=deps.pop("suffixes"),
                generated_dir=generated_dir,
                docker_package=docker_package,
                python_version=pyver,
                bentoml_version=bentoml_version,
                architectures=deps.pop("architectures"),
                templates_dir=deps.pop("templates_dir"),
            )
            build_ctxs[f"{oses}_python{pyver}"] = BuildCtx(
                base_image=deps.pop("base_image"),
                header=deps.pop("header"),
                envars=deps.pop("envars"),
                cuda_ctx=cuda_ctx,
                shared_ctx=shared_ctx,
                **deps,
            )
            release_ctxs[f"{oses}_python{pyver}"] = ReleaseCtx(
                shared_ctx=shared_ctx, **deps
            )
        release_os[release_string] = los
    return build_ctxs, release_ctxs, release_os
