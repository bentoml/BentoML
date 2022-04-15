from __future__ import annotations

import os
import typing as t
from typing import TYPE_CHECKING
from pathlib import Path
from collections import defaultdict
from ._configuration import DockerManagerContainer

import fs

from ._utils import render_template

DOCKER_DIRECTORY = Path(os.path.dirname(__file__)).parent.parent

if TYPE_CHECKING:
    from fs.base import FS

    IncludeMapping = t.Dict[str, str]

CUDA_ARCHITECTURE_PER_DISTRO = {
    "debian11": ["amd64", "arm64v8"],
    "debian10": ["amd64", "arm64v8"],
    "ubi8": ["amd64", "arm64v8", "ppc64le"],
}

ARCHITECTURE_PER_DISTRO = {
    "debian11": DockerManagerContainer.SUPPORTED_ARCHITECTURE_TYPE,
    "debian10": DockerManagerContainer.SUPPORTED_ARCHITECTURE_TYPE,
    "ubi8": DockerManagerContainer.SUPPORTED_ARCHITECTURE_TYPE,
    "alpine3.14": ["amd64", "arm64v8", "ppc64le", "s390x"],
    "amazonlinux2": ["amd64", "arm64v8"],
}


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
