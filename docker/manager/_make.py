import typing as t
import logging
from copy import deepcopy
from typing import TYPE_CHECKING
from itertools import product
from collections import defaultdict

import cattr
from manager._utils import ctx_unstructure_hook
from manager._utils import inject_deepcopy_args
from manager._schemas import CUDA
from manager._schemas import BuildCtx
from manager._schemas import SharedCtx
from manager._schemas import ReleaseCtx
from manager._schemas import CUDAVersion
from manager._schemas import generate_cuda_context
from manager._configuration import RELEASE_PREFIX

if TYPE_CHECKING:
    from manager._click_utils import Environment

logger = logging.getLogger(__name__)


__all__ = ["set_generation_context"]


@inject_deepcopy_args
def create_context(
    distro_info: t.Dict[str, t.Any],
    distro_name: str,
    python_version: str,
    release_string: str,
    ctx: "Environment",
) -> t.Tuple[BuildCtx, ReleaseCtx]:
    cuda_version_full = release_string[len(RELEASE_PREFIX) :]
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
        cuda_url_suffix = dependencies.get("cuda_url_suffixes", None)
        supported_arch = generate_cuda_context(dependencies=dependencies)
        cuda_ctx = CUDA(
            cuda_suffix_url=cuda_url_suffix,
            version=cuda_version_,
            amd64=supported_arch.get("amd64", None),
            arm64v8=supported_arch.get("arm64v8", None),
            ppc64le=supported_arch.get("ppc64le", None),
        )
    except AttributeError:
        cuda_ctx = None

    shared_ctx = SharedCtx(
        distro_name=distro_name,
        python_version=python_version,
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
        cuda_ctx=cuda_ctx,
        shared_ctx=shared_ctx,
    )
    release_ctx = ReleaseCtx(shared_ctx=shared_ctx, fs_=ctx._templates_dir)
    return build_ctx, release_ctx


cattr.register_unstructure_hook(BuildCtx, ctx_unstructure_hook)
cattr.register_unstructure_hook(ReleaseCtx, ctx_unstructure_hook)


def set_generation_context(ctx: "Environment", contexts: t.Dict[str, t.Any]) -> None:
    release_ctx, build_ctx = defaultdict(list), defaultdict(list)

    for pyver, (release_string, distros) in product(
        ctx.python_version, contexts.items()
    ):
        for distro_name, distro in distros.items():
            _build_ctx, _release_ctx = create_context(
                ctx=ctx,
                distro_info=deepcopy(distro),
                python_version=pyver,
                distro_name=distro_name,
                release_string=release_string,
            )
            build_ctx[distro_name].append(_build_ctx)
            release_ctx[distro_name].append(_release_ctx)

    ctx.build_ctx, ctx.release_ctx = build_ctx, release_ctx
