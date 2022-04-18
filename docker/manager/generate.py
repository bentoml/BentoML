from __future__ import annotations
from collections import defaultdict

import typing as t
import logging
import itertools

import fs
import cattrs

from ._make import (
    DockerfileGenerationContext,
    generate_releases_tags_mapping,
    DistrosManifest,
)
from ._utils import render_template
from ._configuration import get_manifest_info
from ._configuration import DockerManagerContainer

logger = logging.getLogger(__name__)


def generate_dockerfiles():
    """
    Generate Dockerfile for a given docker package.
    """

    cuda_url_target_arch = {
        "amd64": "x86_64",
        "arm64v8": "sbsa",
        "ppc64le": "ppc64le",
    }
    for distros, context in get_manifest_info().items():
        architectures, release_types = (
            context["architectures"],
            context["release_types"],
        )
        for (architecture, release_type) in itertools.product(
            architectures, release_types
        ):
            templates_dir = context["templates_dir"]
            base_image = context["base_image"]
            if templates_dir == "rhel" and "ubi" in base_image:
                base_image = base_image.format(f"$UBIFORMAT")

            file_context = DockerfileGenerationContext(
                distros=distros,
                release_type=release_type,
                architecture=architecture,
                base_image=base_image,
                templates_dir=templates_dir,
                suffixes=context["suffixes"],
                conda=context["conda"],
                header=context["header"],
                envars=context["envars"],
                cuda=context.get("dependencies", None),
            )
            for input_path, output_path in file_context.paths_mapping.items():
                if "conda" in input_path:
                    base_tag = file_context.base_tags["conda"]
                else:
                    base_tag = file_context.base_tags["base"]

                render_template(
                    input_path,
                    output_path,
                    base_tag=base_tag,
                    architectures=architectures,
                    cuda_url=cuda_url_target_arch[architecture]
                    if "rhel" in input_path
                    else None,
                    organization=DockerManagerContainer.organization,
                    xx_image="tonistiigi/xx",
                    xx_version="1.1.0",
                    **cattrs.unstructure(file_context),
                )


def get_python_version_from_tag(tag: str) -> t.List[int]:
    return [int(u) for u in tag.split(":")[-1].split("-")[1].strip("python").split(".")]


def generate_readmes():
    tag_ref = defaultdict(list)
    arch = {}

    for distro, context in get_manifest_info().items():
        ctx = DistrosManifest(**context)
        results = generate_releases_tags_mapping(distro, ctx)

        def result_sort_key(result):
            return get_python_version_from_tag(result[0])

        tag_ref[distro] = sorted(results, key=result_sort_key)
        arch[distro] = ctx.architectures

    readme_jinja2_context = {
        "bentoml_package": DockerManagerContainer.docker_package,
        "bentoml_release_version": DockerManagerContainer.bentoml_version.get(),
        "supported": arch,
        "tag_ref": tag_ref,
        "emphemeral": False,
    }

    def render_from_components(
        output_path: str,
        *,
        extends: t.Optional[t.Dict[str, t.Any]] = None,
    ):
        if extends:
            readme_jinja2_context.update(extends)
        readme_tmpl = fs.path.combine("docs", "README.md.j2")
        tmp_fs = fs.open_fs("temp://")

        component_files = ["headers.md.j2", "body.md.j2"]
        final_context = {}
        for f in component_files:
            tmpl = fs.path.join("docs", "components", f)
            filename = f.strip(".j2")
            render_template(
                tmpl, f"/{filename}", out_fs=tmp_fs, **readme_jinja2_context
            )
            with tmp_fs.open(filename, "r") as outf:
                final_context[filename.split(".")[0]] = "".join(outf.readlines())
        render_template(readme_tmpl, output_path, **final_context)

    render_from_components(f"/{DockerManagerContainer.docker_package}/README.md")
    render_from_components(f"/README.md", extends={"emphemeral": True})
