import typing as t
from copy import deepcopy
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from manager._types import GenericNestedDict


# CUDA-related
CUDA_REPO_URL = "https://developer.download.nvidia.com/compute/cuda/repos/{suffixes}"
ML_REPO_URL = (
    "https://developer.download.nvidia.com/compute/machine-learning/repos/{suffixes}"
)


@attrs.define
class LibraryCtx:
    version: t.Optional[str] = attrs.field(default="")
    major_version: t.Optional[str] = attrs.field(default="")


@attrs.define
class CUDAVersion:
    major: str
    minor: str
    patch: str
    shortened: str
    full: str


@attrs.define
class CUDASupportedArch:
    cudnn: LibraryCtx
    cuda_repo_url: str
    ml_repo_url: str
    requires: str
    components: t.Dict[str, LibraryCtx] = attrs.field(default=attrs.Factory(dict))


@attrs.define
class CUDACtx:
    version: CUDAVersion
    dependencies: "t.Optional[GenericNestedDict]" = attrs.field(default=None)
    supported_architecture: t.Optional[t.Dict[str, CUDASupportedArch]] = attrs.field(
        default=None
    )

    def __attrs_post_init__(self):
        if self.dependencies is not None:
            generate_cuda_context(self, self.dependencies)
            object.__setattr__(self, "dependencies", None)


def generate_cuda_context(self: CUDACtx, deps_per_arch: "GenericNestedDict") -> None:
    obj = deepcopy(deps_per_arch)
    cuda_req: "GenericNestedDict" = obj.get("cuda")  # type: ignore
    cuda_url_suffixes = obj.pop("cuda_url_suffixes")

    cuda_supported_arch = {}
    for arch, requirements in cuda_req.items():
        components = requirements["components"]
        requires = requirements.pop("requires")

        def get_cudnn_version() -> LibraryCtx:
            for k, v in components.items():
                if "cudnn" in k:
                    components.pop(k)
                    return LibraryCtx(
                        version=v["version"], major_version=v["version"][:1]
                    )
            return LibraryCtx()

        cudnn_context = get_cudnn_version()
        cuda_supported_arch[arch] = CUDASupportedArch(
            cudnn=cudnn_context,
            components={
                k: LibraryCtx(version=v["version"]) for k, v in components.items()
            },
            cuda_repo_url=CUDA_REPO_URL.format(suffixes=cuda_url_suffixes),
            ml_repo_url=ML_REPO_URL.format(suffixes=cuda_url_suffixes),
            requires=requires,
        )
    self.supported_architecture = cuda_supported_arch
