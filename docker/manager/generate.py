import typing as t
import logging
from functools import partial
from collections import defaultdict

import fs
import cattr
import click

from ._internal._make import CUDA
from ._internal.utils import send_log
from ._internal.utils import render_template
from ._internal.groups import Environment
from ._internal.groups import pass_environment
from ._internal._gen_manifest import gen_manifest

logger = logging.getLogger(__name__)


def add_generation_command(cli: click.Group) -> None:
    @cli.command(name="create-manifest")
    @pass_environment
    def create_manifest(ctx: Environment) -> None:
        """
        Generate a manifest files to edit.
        Note that we still need to customize this manifest files to fit with our usecase.
        """
        gen_manifest(
            ctx.docker_package,
            ctx.cuda_version,
            ctx.distros,
            overwrite=ctx.overwrite,
            registries=ctx.registries,
            docker_fs=ctx._fs,
        )

    @cli.command()
    @pass_environment
    def generate(ctx: Environment) -> None:
        """
        Generate Dockerfile and README for a given docker package.

        \b
        Usage:
            manager generate bento-server --bentoml_version 1.0.0a6
        """

        try:
            if ctx.overwrite:
                ctx._generated_dir.removetree(".")

            # generate readmes and dockerfiles
            generate_dockerfiles(ctx)
            generate_readmes(ctx)
            if ctx.verbose:
                send_log(
                    f"[green]Finished generating {ctx.docker_package}...[/]",
                    extra={"markup": True},
                )
        finally:
            if ctx.verbose:
                send_log(
                    "[bold yellow]Dump context to [bold]generated[/]... [/]",
                    extra={"markup": True},
                    _manager_level=logging.DEBUG,
                )


def generate_dockerfiles(ctx: Environment):

    generated_files = []
    for build_info, tag_info in zip(ctx.build_ctx.values(), ctx.release_ctx.values()):

        for i in range(len(build_info)):
            bi, ti = build_info[i], tag_info[i]

            shared_ctx = bi.shared_ctx

            cuda_context = bi.cuda_ctx
            if isinstance(cuda_context, CUDA):
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
            serialize_cuda = cattr.unstructure(cuda)
            arch_ctx = shared_ctx.architectures

            metadata = {
                "header": bi.header,
                "base_image": bi.base_image,
                "envars": bi.envars,
                "package": f"{ctx.organization}/{shared_ctx.docker_package}",
                "python_version": shared_ctx.python_version,
                "arch_ctx": arch_ctx,
            }

            cuda_target_arch = {
                "amd64": "x86_64",
                "arm64v8": "sbsa",
                "ppc64le": "ppc64le",
            }

            for paths in ti.release_tags.values():
                generated_output = ctx._generated_dir.getsyspath(paths["output_path"])
                if generated_output in generated_files:
                    continue
                else:
                    generated_files.append(generated_output)
                    for tpl_file in paths["input_paths"]:
                        to_render = partial(
                            render_template,
                            input_name=tpl_file,
                            inp_fs=ctx._templates_dir,
                            output_path=paths["output_path"],
                            out_fs=ctx._generated_dir,
                            build_tag=paths["build_tag"],
                            cuda=serialize_cuda,
                            metadata=metadata,
                            xx_image="tonistiigi/xx",
                            xx_version="1.1.0",
                        )

                        try:
                            if "rhel" in shared_ctx.templates_dir:
                                cuda_supported_arch = [
                                    k for k in arch_ctx if hasattr(cuda_context, k)
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


def generate_readmes(ctx: Environment) -> None:
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
        arch[distro] = distro_info[0].shared_ctx.architectures

    readme_context = {
        "bentoml_package": ctx.docker_package,
        "bentoml_release_version": ctx.bentoml_version,
        "supported": arch,
        "tag_ref": tag_ref,
        "emphemeral": False,
    }

    readme_file = fs.path.combine("docs", "README.md.j2")

    render_template(
        readme_file,
        ctx._templates_dir,
        ctx.docker_package,
        ctx._generated_dir,
        **readme_context,
    )

    readme_context["emphemeral"] = True

    render_template(
        readme_file,
        ctx._templates_dir,
        "/",
        ctx._generated_dir,
        **readme_context,
    )


