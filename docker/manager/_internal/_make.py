from __future__ import annotations

import typing as t
import logging
from copy import deepcopy
from typing import TYPE_CHECKING
from itertools import product
from collections import defaultdict

import fs
import attrs
import attrs.converters
from fs.base import FS

from ._funcs import send_log
from ._funcs import inject_deepcopy_args
from ._configuration import DOCKERFILE_BUILD_HIERARCHY
from ._configuration import SUPPORTED_ARCHITECTURE_TYPE
from ._configuration import DOCKER_TARGETARCH_LINUX_UNAME_ARCH_MAPPING

if TYPE_CHECKING:
    from .groups import Environment

    GenericDict = t.Dict[str, t.Any]

logger = logging.getLogger(__name__)


__all__ = ["set_generation_context"]


# CUDA-related
CUDA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{suffixes}"
ML_REPO_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/repos/{suffixes}"
)


@attrs.define
class LibraryVersion:
    version: str
    major_version: str = attrs.field(
        default=attrs.Factory(lambda self: self.version.split(".")[0], takes_self=True)
    )


@attrs.define
class CUDAVersion:
    major: str
    minor: str
    patch: str
    full: str
    shortened: str


@attrs.define
class SupportedArchitecture:
    requires: str
    cudnn: t.Optional[LibraryVersion]
    components: t.Dict[str, LibraryVersion] = attrs.field(factory=dict)


@attrs.define
class CUDACtx:
    cuda_suffix_url: str
    version: CUDAVersion
    amd64: t.Optional[SupportedArchitecture]
    arm64v8: t.Optional[SupportedArchitecture]
    ppc64le: t.Optional[SupportedArchitecture]
    cuda_repo_url: str = attrs.field(init=False)
    ml_repo_url: str = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.cuda_repo_url = CUDA_REPO_URL.format(suffixes=self.cuda_suffix_url)
        self.ml_repo_url = ML_REPO_URL.format(suffixes=self.cuda_suffix_url)


@inject_deepcopy_args
def generate_cuda_context(
    dependencies: t.Dict[str, GenericDict],
) -> t.Tuple[t.Dict[str, SupportedArchitecture], str]:
    cuda_req = dependencies.get("cuda", None)  # type: ignore

    cuda_supported_arch = {}
    suffix = ""
    if cuda_req is not None:
        suffix = cuda_req.pop("cuda_url_suffixes")
        for arch, requirements in cuda_req.items():
            components = requirements["components"]
            requires = requirements["requires"]

            def get_cudnn_version() -> t.Optional[LibraryVersion]:
                for key in components:
                    if key.startswith("cudnn"):
                        cudnn = components.pop(key)
                        return LibraryVersion(
                            version=cudnn["version"], major_version=key[-1]
                        )
                send_log(
                    "current CUDA components doesn't support CUDNN.",
                    _manager_level=logging.DEBUG,
                )

            cudnn_context = get_cudnn_version()

            cuda_supported_arch[arch] = SupportedArchitecture(
                cudnn=cudnn_context,
                components={
                    k: LibraryVersion(version=v["version"])
                    for k, v in components.items()
                },
                requires=requires,
            )
    return cuda_supported_arch, suffix


@attrs.define
class SharedCtx:
    suffixes: str
    distro_name: str
    python_version: str
    bentoml_version: str

    docker_package: str
    templates_dir: str
    conda: bool

    release_types: t.List[str] = attrs.field(
        factory=list,
        validator=lambda _, __, value: set(value).issubset(DOCKERFILE_BUILD_HIERARCHY),
    )
    architectures: t.List[str] = attrs.field(
        factory=list,
        validator=[
            lambda _, __, value: set(value).issubset(
                DOCKER_TARGETARCH_LINUX_UNAME_ARCH_MAPPING
            ),
            lambda _, __, value: set(value).issubset(SUPPORTED_ARCHITECTURE_TYPE),
        ],
    )


def update_envars_dict(
    args: "GenericDict", updater: t.Union[str, GenericDict, t.List[str]]
) -> "GenericDict":
    # Args will have format ARG=foobar
    if isinstance(updater, str):
        arg, _, value = updater.partition("=")
        args[arg] = value
    elif isinstance(updater, list):
        for v in updater:
            update_envars_dict(args, v)
    elif isinstance(updater, dict):
        for arg, value in updater.items():
            args[arg] = value
    else:
        send_log(
            f"cannot add to args dict with unknown type {type(updater)}",
            _manager_level=logging.ERROR,
        )
    return args


@attrs.define
class BuildCtx:
    header: str
    base_image: str
    cuda_context: t.Optional[CUDACtx]
    shared_context: SharedCtx
    envars: t.Dict[str, str]

    def __attrs_post_init__(self):
        args = {
            "BENTOML_VERSION": self.shared_context.bentoml_version,
            "PYTHON_VERSION": self.shared_context.python_version,
        }
        self.envars = update_envars_dict(args, self.envars)


def create_tags_per_distros(
    shared_context: SharedCtx,
) -> t.Dict[str, t.Dict[str, t.Union[str, t.List[str]]]]:
    # returns a list of strings following the build hierarchy
    TAG_FORMAT = "{release_type}-python{python_version}-{suffixes}"

    metadata = {}
    for rt in shared_context.release_types:
        tag = []
        image_tag = (
            shared_context.docker_package
            + ":"
            + "-".join(
                [
                    TAG_FORMAT.format(
                        release_type=shared_context.bentoml_version,
                        python_version=shared_context.python_version,
                        suffixes=shared_context.suffixes,
                    ),
                    rt,
                ]
            )
        )
        if rt not in ["runtime", "cudnn"]:
            image_tag = (
                shared_context.docker_package
                + ":"
                + TAG_FORMAT.format(
                    release_type=rt,
                    python_version=shared_context.python_version,
                    suffixes=shared_context.suffixes,
                )
            )
        tag.append(image_tag)
        if shared_context.conda:
            conda_tag = f"{image_tag}-conda"
            tag.append(conda_tag)

        if rt != "base":
            build_tag = TAG_FORMAT.format(
                release_type="base",
                python_version="$PYTHON_VERSION",
                suffixes=shared_context.suffixes,
            )
        else:
            build_tag = ""
        metadata[rt] = {"image_tag": image_tag, "build_tag": build_tag}

    return metadata


def create_path_context(shared_context: SharedCtx, fs_: FS) -> GenericDict:

    tags = create_tags_per_distros(shared_context)

    release_tag = {}

    for release_type in shared_context.release_types:
        tag = tags[release_type]

        for image_tag in tag["image_tag"]:

            output_path = fs.path.join(
                shared_context.docker_package,
                shared_context.distro_name,
                release_type,
            )

            dockerfile = "Dockerfile"
            filters = [f"{release_type}-*.j2"]
            if shared_context.conda:
                dockerfile += "-conda"
                filters = [f"conda-{release_type}-*.j2"]

            git_tree_path = fs.path.combine(output_path, dockerfile)

            release_tag[image_tag] = {
                "output_path": output_path,
                "input_paths": [
                    f
                    for f in fs_.walk.files(filter=filters)
                    if shared_context.templates_dir in f
                ],
                "git_tree_path": git_tree_path,
                "build_tag": tag["build_tag"],
            }
    return release_tag


@attrs.define
class ReleaseCtx:
    fs: FS
    shared_context: SharedCtx
    release_tags: t.Dict[str, t.Any] = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.release_tags = create_path_context(self.shared_context, self.fs)


@inject_deepcopy_args
def create_distro_context(
    distro_info: t.Dict[str, t.Any],
    distro_name: str,
    python_version: str,
    release_string: str,
    ctx: "Environment",
) -> t.Tuple[BuildCtx, ReleaseCtx]:
    cuda_version_full = release_string[len("w_cuda_v") :]
    major, minor, patch = cuda_version_full.split(".")
    cuda_version_ = CUDAVersion(
        major=major,
        minor=minor,
        patch=patch,
        shortened=f"{major}.{minor}",
        full=cuda_version_full,
    )

    try:
        dependencies = distro_info.pop("dependencies")
        supported_arch, cuda_url_suffix = generate_cuda_context(
            dependencies=dependencies
        )
        cuda_context = CUDACtx(
            cuda_suffix_url=cuda_url_suffix,
            version=cuda_version_,
            amd64=supported_arch.get("amd64", None),
            arm64v8=supported_arch.get("arm64v8", None),
            ppc64le=supported_arch.get("ppc64le", None),
        )
    except AttributeError:
        cuda_context = None

    shared_context = SharedCtx(
        distro_name=distro_name,
        python_version=python_version,
        conda=distro_info.pop("conda"),
        suffixes=distro_info.pop("suffixes"),
        release_types=distro_info.pop("release_types"),
        templates_dir=distro_info.pop("templates_dir"),
        architectures=distro_info.pop("architectures"),
        bentoml_version=ctx.bentoml_version,
        docker_package=ctx.docker_package,
    )
    build_ctx = BuildCtx(
        base_image=distro_info.pop("base_image"),
        header=distro_info.pop("header"),
        envars=distro_info.pop("envars"),
        cuda_context=cuda_context,
        shared_context=shared_context,
    )
    release_ctx = ReleaseCtx(shared_context=shared_context, fs=ctx._templates_dir)
    return build_ctx, release_ctx


def set_generation_context(ctx: "Environment", distros_info: GenericDict) -> None:
    release_ctx, build_ctx = defaultdict(list), defaultdict(list)

    for pyver, (release_string, distros) in product(
        ctx.python_version, distros_info.items()
    ):
        for distro_name, distro in distros.items():
            _build_ctx, _release_ctx = create_distro_context(
                ctx=ctx,
                distro_info=deepcopy(distro),
                python_version=pyver,
                distro_name=distro_name,
                release_string=release_string,
            )
            build_ctx[distro_name].append(_build_ctx)
            release_ctx[distro_name].append(_release_ctx)

    ctx.build_ctx, ctx.release_ctx = build_ctx, release_ctx
