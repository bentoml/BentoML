from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING
import logging
import itertools
from collections import defaultdict

import fs
import cattrs
from simple_di import inject
from manager._make import DistrosManifest
from manager._make import DockerfileGenerationContext
from manager._make import generate_releases_tags_mapping
from manager._utils import send_log
from manager._utils import render_template
from manager._configuration import get_manifest_info
from manager._configuration import DockerManagerContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from argparse import Namespace


def generate_dockerfiles(args: Namespace) -> None:
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
                    organization=args.organization,
                    xx_image="tonistiigi/xx",
                    xx_version="1.1.0",
                    **cattrs.unstructure(file_context),
                )


def get_python_version_from_tag(tag: str) -> t.List[int]:
    return [int(u) for u in tag.split(":")[-1].split("-")[1].strip("python").split(".")]


@inject
def generate_readmes(
    args: Namespace,
    *,
    docker_package: str = DockerManagerContainer.docker_package,
) -> None:
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
        "bentoml_package": docker_package,
        "bentoml_release_version": args.bentoml_version,
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

    render_from_components(f"/{docker_package}/README.md")
    render_from_components(f"/README.md", extends={"emphemeral": True})


def entrypoint(args: Namespace) -> None:
    generate_dockerfiles(args)
    generate_readmes(args)

    send_log(
        "[bold green]Generated Dockerfiles and README.md.[/]", extra={"markup": True}
    )
