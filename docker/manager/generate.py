import os
import json
import typing as t
import logging
from typing import TYPE_CHECKING
from pathlib import Path
from functools import partial
from collections import defaultdict

import click
from jinja2 import Environment
from manager import SUPPORTED_PYTHON_VERSION
from manager._utils import as_posix
from manager._utils import serialize_to_json
from manager._context import BuildCtx
from manager._context import EXTENSION
from manager._context import ReleaseCtx
from manager._context import load_context
from manager._context import RELEASE_PREFIX
from manager._context import DOCKERFILE_NAME
from manager._container import ManagerContainer
from manager.exceptions import ManagerException

if TYPE_CHECKING:
    from manager._types import GenericDict

logger = logging.getLogger(__name__)


README_TEMPLATE = ManagerContainer.template_dir.joinpath("docs", "README.md.j2")


def add_generation_command(cli: click.Group) -> None:
    @cli.command()
    @click.option(
        "--bentoml-version",
        required=True,
        type=click.STRING,
        help="targeted bentoml version",
    )
    @click.option(
        "--dump-metadata",
        is_flag=True,
        default=False,
        help="dump metadata to files. Useful for debugging",
    )
    @click.option(
        "--python-version",
        required=False,
        type=click.Choice(SUPPORTED_PYTHON_VERSION),
        multiple=True,
        help=f"Targets a python version, default to {SUPPORTED_PYTHON_VERSION}",
        default=SUPPORTED_PYTHON_VERSION,
    )
    @click.option(
        "--generated-dir",
        type=click.STRING,
        metavar="generated",
        help=f"Output directory for generated Dockerfile, default to {ManagerContainer.generated_dir.as_posix()}",
        default=as_posix(ManagerContainer.generated_dir),
    )
    def generate(
        docker_package: str,
        bentoml_version: str,
        generated_dir: str,
        dump_metadata: bool,
        python_version: t.Tuple[str],
    ) -> None:
        """
        Generate Dockerfile and README for a given docker package.

        \b
        Usage:
            manager generate bento-server --bentoml_version 1.0.0a6
            manager generate bento-server --bentoml_version 1.0.0a6 --dump-metadata

        \b
        NOTE: --dump_metadata is useful when development to see build and releases context.

        """
        if os.geteuid() == 0:
            raise ManagerException(
                "`generate` shouldn't be running as root, as "
                "wrong file permission would break the workflow."
            )
        build_ctx, release_ctx, os_list = load_context(
            docker_package=docker_package,
            bentoml_version=bentoml_version,
            python_version=python_version,
            generated_dir=generated_dir,
        )

        if dump_metadata:
            with ManagerContainer.generated_dir.joinpath("build.meta.json").open(
                "w"
            ) as ouf1, ManagerContainer.generated_dir.joinpath(
                "releases.meta.json"
            ).open(
                "w"
            ) as ouf2:
                ouf1.write(json.dumps(serialize_to_json(build_ctx), indent=2))
                ouf2.write(json.dumps(serialize_to_json(release_ctx), indent=2))

        # generate readmes and dockerfiles
        _generate_readmes(
            docker_package,
            bentoml_version,
            release_ctx=release_ctx,
            os_list=os_list,
        )
        _generate_dockerfiles(release_ctx=release_ctx, build_ctx=build_ctx)


def _generate_readmes(
    package: str,
    bentoml_version: str,
    release_ctx: "t.Dict[str, ReleaseCtx]",
    os_list: t.Dict[str, t.List[str]],
) -> None:
    supported_architecture = {}
    sub_v = []
    tag_ref = defaultdict(list)
    for version in os_list:
        sv = version[len(RELEASE_PREFIX) :]
        if sv not in sub_v:
            sub_v.append(sv)
    for rltx in release_ctx.values():
        if supported_architecture.get(rltx.shared_ctx.distro_name) is None:
            supported_architecture[
                rltx.shared_ctx.distro_name
            ] = rltx.shared_ctx.architectures
        tag_ref[rltx.shared_ctx.distro_name].append(
            {
                k: v["git_tree_path"]
                for k, v in rltx.release_tags.items()
                if "base" not in k
            }
        )
    readme_context = {
        "bentoml_package": package,
        "bentoml_release_version": bentoml_version,
        "support_arch": supported_architecture,
        "tag_ref": tag_ref,
        "sub_versions": sub_v,
    }
    _render_template(
        input_path=README_TEMPLATE,
        output_path=ManagerContainer.generated_dir.joinpath(package),
        release_ctx=readme_context,
    )


def _generate_dockerfiles(build_ctx: t.Dict[str, BuildCtx], release_ctx):
    for tags_ctx, rls_ctx in zip(build_ctx.values(), release_ctx.values()):
        if tags_ctx.cuda_ctx.supported_architecture is not None:
            cuda = {
                "version": tags_ctx.cuda_ctx.version,
                "amd64": tags_ctx.cuda_ctx.supported_architecture["amd64"],
                "arm64v8": tags_ctx.cuda_ctx.supported_architecture["arm64v8"],
            }

            ppc64le = tags_ctx.cuda_ctx.supported_architecture.get("ppc64le", None)
            if ppc64le is not None:
                cuda["ppc64le"] = ppc64le
            cuda = serialize_to_json(cuda)
        else:
            cuda = None
        generated = []
        metadata = {
            "header": tags_ctx.header,
            "base_image": tags_ctx.base_image,
            "envars": tags_ctx.envars,
            "package": tags_ctx.shared_ctx.docker_package,
            "arch_ctx": tags_ctx.shared_ctx.architectures,
        }

        for img_tag, paths in rls_ctx.release_tags.items():
            for tpl_file in paths["input_paths"]:
                logger.info(
                    f":brick: [yellow]Generating[/] [magenta]{tpl_file.split('/')[-1]}[/] for [blue]{img_tag}[/blue] ...",
                    extra={"markup": True},
                )
                if tpl_file in generated:
                    continue
                else:
                    output_path = Path(paths["output_path"])
                    to_render = partial(
                        _render_template,
                        input_path=Path(tpl_file),
                        output_path=output_path,
                        build_tag=paths["build_tag"],
                        cuda=cuda,
                        metadata=metadata,
                    )
                    if "rhel" in tags_ctx.shared_ctx.templates_dir:
                        arch_ctx = tags_ctx.shared_ctx.architectures
                        if tags_ctx.cuda_ctx.supported_architecture is not None:
                            cuda_supported_arch = [
                                k
                                for k in arch_ctx
                                if k in tags_ctx.cuda_ctx.supported_architecture
                            ]
                            for sa in cuda_supported_arch:
                                to_render(arch=sa)
                    else:
                        to_render()
                    generated.append(tpl_file)


def _render_template(
    input_path: Path,
    output_path: Path,
    release_ctx: "t.Optional[GenericDict]" = None,
    build_ctx: "t.Optional[GenericDict]" = None,
    arch: t.Optional[str] = None,
    build_tag: t.Optional[str] = None,
    **kwargs: t.Any,
):
    """
    Render .j2 templates to output path
    Args:
        input_path (:obj:`pathlib.Path`):
            t.List of input path
        output_path (:obj:`pathlib.Path`):
            Output path
        metadata (:obj:`t.Dict[str, Union[str, t.List[str], t.Dict[str, ...]]]`):
            templates context
        build_tag (:obj:`t.Optional[str]`):
            strictly use for FROM args for base image.
    """
    cuda_target_arch = {"amd64": "x86_64", "arm64v8": "sbsa", "ppc64le": "ppc64le"}
    template_env = Environment(
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        trim_blocks=True,
        lstrip_blocks=True,
    )
    input_name = input_path.name
    if "readme".upper() in input_name:
        output_name = input_path.stem
    elif "dockerfile" in input_name:
        output_name = DOCKERFILE_NAME
    else:
        # specific cases for rhel
        output_name = (
            input_name[len("cudnn-") : -len(EXTENSION)]
            if input_name.startswith("cudnn-") and input_name.endswith(EXTENSION)
            else input_name
        )
        if arch:
            output_name += f"-{arch}"
    if not output_path.exists():
        logger.debug(f"Creating {as_posix(output_path)}...")
        output_path.mkdir(parents=True, exist_ok=True)

    with input_path.open("r") as inf:
        template = template_env.from_string(inf.read())

    rendered_path = output_path.joinpath(output_name)

    with rendered_path.open("w") as ouf:
        ouf.write(
            template.render(
                build_ctx=build_ctx,
                release_ctx=release_ctx,
                build_tag=build_tag,
                arch=arch,
                cuda_target_arch=cuda_target_arch,
                **kwargs,
            )
        )
