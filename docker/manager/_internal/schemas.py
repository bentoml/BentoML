import typing as t
import logging
from typing import TYPE_CHECKING

import fs
import attrs
import cattr
import attrs.converters
from fs.base import FS

from .utils import send_log
from .utils import DOCKERFILE_NAME
from .utils import raise_exception
from .utils import ctx_unstructure_hook
from .utils import inject_deepcopy_args
from .utils import DOCKERFILE_BUILD_HIERARCHY
from .utils import SUPPORTED_ARCHITECTURE_TYPE
from .exceptions import ManagerException

if TYPE_CHECKING:
    from .types import StrList
    from .types import GenericDict
    from .types import DoubleNestedDict
    from .types import GenericNestedDict


logger = logging.getLogger(__name__)

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


# CUDA-related
CUDA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{suffixes}"
ML_REPO_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/repos/{suffixes}"
)
CUDA_SUPPORTED_ARCH = ["amd64", "arm64v8", "ppc64le"]


@attrs.define
class LibraryVersion:
    version: str
    major_version: str = attrs.field(
        default=attrs.Factory(lambda self: self.version[:1], takes_self=True)
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
class CUDA:
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
    dependencies: "GenericNestedDict",
) -> t.Dict[str, SupportedArchitecture]:
    cuda_req: "GenericNestedDict" = dependencies.get("cuda", None)  # type: ignore

    cuda_supported_arch = {}
    if cuda_req is not None:
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
    return cuda_supported_arch


cattr.register_unstructure_hook(CUDA, ctx_unstructure_hook)
cattr.register_unstructure_hook(SupportedArchitecture, ctx_unstructure_hook)
cattr.register_unstructure_hook(LibraryVersion, ctx_unstructure_hook)
cattr.register_unstructure_hook(CUDAVersion, ctx_unstructure_hook)


def is_supported_by_docker(instance: t.Any, attribute: t.Any, value: t.List[str]):
    for v in value:
        if v not in ALL_DOCKER_SUPPORTED_ARCHITECTURE:
            raise ManagerException(f"{v} architecture is not supported by Docker.")
        if v not in SUPPORTED_ARCHITECTURE_TYPE:
            send_log(
                f"{v} is not yet supported by BentoML. Use with care!",
                _manager_level=logging.DEBUG,
            )


def is_in_build_hierachy(instance: t.Any, attribute: t.Any, value: t.List[str]):
    for v in value:
        if v not in DOCKERFILE_BUILD_HIERARCHY:
            raise ManagerException(
                f"{v} is not in DOCKERFILE_BUILD_HIERARCHY. If adding a new type of images make sure to update DOCKERFILE_BUILD_HIERARCHY."
            )


@attrs.define
class SharedCtx:
    suffixes: str
    distro_name: str
    python_version: str
    bentoml_version: str

    docker_package: str
    templates_dir: str

    release_types: t.List[str] = attrs.field(
        factory=list, validator=is_in_build_hierachy
    )
    architectures: t.List[str] = attrs.field(
        factory=list, validator=is_supported_by_docker
    )


@raise_exception
def update_envars_dict(
    args: "GenericDict", updater: "t.Union[str, GenericDict, StrList]"
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
    envars: t.Dict[str, str]

    shared_ctx: SharedCtx
    cuda_ctx: t.Optional[CUDA] = attrs.field(
        converter=attrs.converters.default_if_none(""), default=None
    )

    def __attrs_post_init__(self):
        args = {
            "BENTOML_VERSION": self.shared_ctx.bentoml_version,
            "PYTHON_VERSION": self.shared_ctx.python_version,
        }
        self.envars = update_envars_dict(args, self.envars)


def create_tags_per_distros(shared_ctx: SharedCtx) -> "DoubleNestedDict[str]":
    # returns a list of strings following the build hierarchy
    TAG_FORMAT = "{release_type}-python{python_version}-{suffixes}"

    metadata = {}
    for rt in shared_ctx.release_types:
        if rt in ["runtime", "cudnn"]:
            image_tag = (
                shared_ctx.docker_package
                + ":"
                + "-".join(
                    [
                        TAG_FORMAT.format(
                            release_type=shared_ctx.bentoml_version,
                            python_version=shared_ctx.python_version,
                            suffixes=shared_ctx.suffixes,
                        ),
                        rt,
                    ]
                )
            )
        else:
            image_tag = (
                shared_ctx.docker_package
                + ":"
                + TAG_FORMAT.format(
                    release_type=rt,
                    python_version=shared_ctx.python_version,
                    suffixes=shared_ctx.suffixes,
                )
            )

        if rt != "base":
            build_tag = TAG_FORMAT.format(
                release_type="base",
                python_version="$PYTHON_VERSION",
                suffixes=shared_ctx.suffixes,
            )
        else:
            build_tag = ""
        metadata[rt] = {"image_tag": image_tag, "build_tag": build_tag}

    return metadata


def create_path_context(
    shared_ctx: SharedCtx, template_fs_: "FS"
) -> "DoubleNestedDict":

    tags = create_tags_per_distros(shared_ctx)

    release_tag = {}

    for release_type in shared_ctx.release_types:
        tag = tags[release_type]

        output_path = fs.path.join(
            shared_ctx.docker_package,
            shared_ctx.distro_name,
            release_type,
        )
        git_tree_path = fs.path.combine(output_path, DOCKERFILE_NAME)

        release_tag[tag["image_tag"]] = {
            "output_path": output_path,
            "input_paths": [
                f
                for f in template_fs_.walk.files(filter=[f"{release_type}-*.j2"])
                if shared_ctx.templates_dir in f
            ],
            "git_tree_path": git_tree_path,
            "build_tag": tag["build_tag"],
        }
    return release_tag


@attrs.define
class ReleaseCtx:
    fs_: "FS"
    shared_ctx: SharedCtx
    release_tags: t.Dict[str, t.Any] = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.release_tags = create_path_context(self.shared_ctx, self.fs_)
