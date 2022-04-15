from __future__ import annotations
import itertools

import logging

from ._utils import render_template
from ._make import FileGenerationContext
from ._configuration import get_manifest_info, DockerManagerContainer
import cattrs

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

            file_context = FileGenerationContext(
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


def generate_readmes():

    ...
