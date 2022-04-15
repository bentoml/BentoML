from __future__ import annotations

import os
import typing as t
import logging
import itertools
from uuid import uuid1
from typing import TYPE_CHECKING
from pathlib import Path
from functools import partial
from functools import lru_cache
from collections import defaultdict

import fs
import yaml
import cattr
from yamlinclude import YamlIncludeConstructor

from ._make import CUDACtx
from ._funcs import send_log
from ._funcs import render_template
from ._configuration import DOCKERFILE_BUILD_HIERARCHY
from ._configuration import SUPPORTED_ARCHITECTURE_TYPE

DOCKER_DIRECTORY = Path(os.path.dirname(__file__)).parent.parent

if TYPE_CHECKING:
    from fs.base import FS

    from .groups import Environment

    IncludeMapping = t.Dict[str, str]

YamlIncludeConstructor.add_to_loader_class(
    loader_class=yaml.FullLoader,
    base_dir=DOCKER_DIRECTORY.joinpath("manager", "_internal").__fspath__(),
)


CUDA_ARCHITECTURE_PER_DISTRO = {
    "debian11": ["amd64", "arm64v8"],
    "debian10": ["amd64", "arm64v8"],
    "ubi8": ["amd64", "arm64v8", "ppc64le"],
}

ARCHITECTURE_PER_DISTRO = {
    "alpine3.14": SUPPORTED_ARCHITECTURE_TYPE,
    "debian11": SUPPORTED_ARCHITECTURE_TYPE,
    "debian10": SUPPORTED_ARCHITECTURE_TYPE,
    "ubi8": SUPPORTED_ARCHITECTURE_TYPE,
    "amazonlinux2": ["amd64", "arm64v8"],
}


def walk_include_dir(
    include_fs: FS,
    templates_dir: str,
    checker: t.Iterable[t.Any],
    filter_key: t.Callable[[str], t.Any] = lambda x: x,
) -> IncludeMapping:
    if isinstance(checker, str):
        checker = [checker]

    results = {}

    templates_fs = include_fs.makedirs(templates_dir, recreate=True)
    for check, p in itertools.product(
        checker, templates_fs.walk.files(filter=["*.yaml"])
    ):
        if check in p:
            p = p.strip("/").strip(".yaml")
            results[filter_key(p)] = f"!include include.d/{templates_dir}/{p}.yaml"
    return results


def unpack_include_content(include_: IncludeMapping, indent: int = 0) -> str:
    mem_fs = fs.open_fs("mem://")
    tmp_file = partial(mem_fs.open, f".{uuid1()}.yml", encoding="utf-8")
    include_string = "\n".join([f"{k}: {v}\n" for k, v in include_.items()])

    with tmp_file("w") as inf:
        yaml.dump(
            yaml.load(include_string.encode("utf-8"), Loader=yaml.FullLoader), inf
        )
    with tmp_file("r") as res:
        indentation = indent * " "
        res = indentation.join(res.readlines())

    mem_fs.close()

    return res


def gen_dockerfiles(ctx: Environment):

    registry = os.environ.get("DOCKER_REGISTRY", None)
    if registry is None:
        raise Exception("Failed to retrieve docker registry from envars.")

    cached = []
    for builds, tags in zip(ctx.build_ctx.values(), ctx.release_ctx.values()):

        for (build_info, tag_info) in zip(builds, tags):

            shared_context = build_info.shared_context
            cuda_context = build_info.cuda_context
            arch_context = shared_context.architectures
            release_tags = tag_info.release_tags

            if isinstance(cuda_context, CUDACtx):
                cuda = {
                    "version": cuda_context.version,
                    "amd64": cuda_context.amd64,
                    "arm64v8": cuda_context.arm64v8,
                    "cuda_repo_url": cuda_context.cuda_repo_url,
                    "ml_repo_url": cuda_context.ml_repo_url,
                }
                if cuda_context.ppc64le is not None:
                    cuda["ppc64le"] = cuda_context.ppc64le
            else:
                cuda = None

            metadata = {
                "header": build_info.header,
                "base_image": build_info.base_image,
                "envars": build_info.envars,
                "organization": ctx.organization,
                "python_version": shared_context.python_version,
                "arch_context": arch_context,
            }

            cuda_target_arch = {
                "amd64": "x86_64",
                "arm64v8": "sbsa",
                "ppc64le": "ppc64le",
            }

            for paths in release_tags.values():
                for tpl_file in paths["input_paths"]:
                    cached_key = f"{build_info.shared_context.distro_name}{ctx._templates_dir.getsyspath(tpl_file)}"
                    if cached_key in cached:
                        continue
                    else:
                        cached.append(cached_key)
                        to_render = partial(
                            render_template,
                            input_name=tpl_file,
                            inp_fs=ctx._templates_dir,
                            output_path=paths["output_path"],
                            out_fs=ctx._generated_dir,
                            build_tag=paths["build_tag"],
                            cuda=cattr.unstructure(cuda),
                            metadata=metadata,
                            distros=build_info.shared_context.distro_name,
                            xx_image="tonistiigi/xx",
                            xx_version="1.1.0",
                            context_path=paths["output_path"],
                        )

                        try:
                            if "rhel" in shared_context.templates_dir:
                                cuda_supported_arch = [
                                    k for k in arch_context if hasattr(cuda_context, k)
                                ]
                                for sa in cuda_supported_arch:
                                    to_render(arch=sa, cuda_url=cuda_target_arch[sa])
                            else:
                                to_render()
                        except Exception as e:  # pylint: disable=broad-except
                            send_log(
                                f"Error while generating Dockerfiles:\n{e}",
                                _manager_level=logging.ERROR,
                            )
                            raise


def get_python_version_from_tag(tag: str) -> t.List[int]:
    return [int(u) for u in tag.split(":")[-1].split("-")[1].strip("python").split(".")]


def gen_readmes(ctx: Environment) -> None:
    tag_ref = defaultdict(list)
    arch = {}

    for distro, distro_info in ctx.release_ctx.items():
        results = [
            (k, v["git_tree_path"])
            for rltx in distro_info
            for k, v in rltx.release_tags.items()
            if "base" not in k
        ]

        (k_sort := [t[0] for t in results]).sort(
            key=lambda tag: get_python_version_from_tag(tag)
        )
        order = {k: v for v, k in enumerate(k_sort)}

        tag_ref[distro] = sorted(results, key=lambda k: order[k[0]])
        arch[distro] = distro_info[0].shared_context.architectures

    readme_context = {
        "bentoml_package": ctx.docker_package,
        "bentoml_release_version": ctx.bentoml_version,
        "supported": arch,
        "tag_ref": tag_ref,
        "emphemeral": False,
        "main_readme": False,
    }

    readme_tmpl = fs.path.combine("docs", "README.md.j2")
    tmp_fs = fs.open_fs("temp://")

    def render_final_readmes(
        out_name: str,
        out_fs: FS,
        *,
        template_files: t.List[str] = ["headers.md.j2", "body.md.j2"],
        extends: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        final_context = {}

        if extends:
            readme_context.update(extends)

        for file in template_files:
            tmpl = fs.path.combine("docs", file)
            filename = file.strip(".j2")
            render_template(
                tmpl,
                ctx._templates_dir,
                "/",
                tmp_fs,
                output_name=filename,
                **readme_context,
            )
            with tmp_fs.open(filename, "r") as f:
                final_context[filename.split(".")[0]] = "".join(f.readlines())
        render_template(
            readme_tmpl,
            ctx._templates_dir,
            out_name,
            out_fs,
            **final_context,
        )

    render_final_readmes(ctx.docker_package, ctx._generated_dir)
    render_final_readmes("/", ctx._fs, extends={"main_readme": True})
    render_final_readmes("/", ctx._generated_dir, extends={"emphemeral": True})
