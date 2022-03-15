import json
import typing as t
import logging
import traceback
from functools import partial
from collections import defaultdict

import fs
import yaml
import cattr
import click
from simple_di import inject
from simple_di import Provide
from manager._make import CUDA
from manager._utils import unstructure
from manager._utils import render_template
from manager._utils import TEMPLATES_DIR_MAPPING
from manager._utils import DOCKERFILE_BUILD_HIERARCHY
from manager._utils import SUPPORTED_ARCHITECTURE_TYPE
from manager._utils import SUPPORTED_ARCHITECTURE_TYPE_PER_DISTRO
from manager._click_utils import Environment
from manager._click_utils import pass_environment
from manager._configuration import MANIFEST_FILENAME
from manager._configuration import DockerManagerContainer

logger = logging.getLogger(__name__)


def add_generation_command(cli: click.Group) -> None:
    @cli.command(name="create-manifest")
    @pass_environment
    @inject
    def create_manifest(
        ctx: Environment, yaml_loader=Provide[DockerManagerContainer.yaml_loader]
    ):
        """
        Generate a manifest files to edit.
        Note that we still need to customize this manifest files to fit with our usecase.
        """

        registry_str = "\n".join(
            [f"{k}: !include include.d/registry/{k}.yaml" for k in ctx.registries]
        ).encode("utf-8")

        mem_fs = fs.open_fs("mem://")

        with mem_fs.open("registry.yml", "w", encoding="utf-8") as f:
            yaml.dump(yaml.load(registry_str, Loader=yaml_loader), f)
        with mem_fs.open("registry.yml", "r", encoding="utf-8") as f:
            content = f.readlines()

        name = MANIFEST_FILENAME.format(ctx.docker_package, ctx.cuda_version)
        include_path = fs.path.join("manager", "include.d")
        include_fs = ctx._fs.makedirs(include_path, recreate=True)

        if not ctx._manifest_dir.exists(name) or ctx.overwrite:
            cuda_req = {}
            for p in include_fs.walk():
                if ctx.cuda_version in p.path:
                    arch = p.path.rsplit(".", maxsplit=2)[1]
                    cuda_req[f"{arch}"] = f"!include {p.path}"

            spec_tmpl = {
                "cuda_version": ctx.cuda_version,
                "architectures": SUPPORTED_ARCHITECTURE_TYPE,
                "release_types": DOCKERFILE_BUILD_HIERARCHY,
                "cuda_architecture_mapping": cuda_req,
                "registries": "".join(content),
                "supported_distros": {
                    k: f"&{k.strip('0123456789.-')}" for k in ctx.distros
                },
                "architecture_per_distros": SUPPORTED_ARCHITECTURE_TYPE_PER_DISTRO,
                "templates_entries": TEMPLATES_DIR_MAPPING,
            }

            render_template(
                "spec.yaml.j2",
                include_fs,
                "/",
                ctx._manifest_dir,
                output_name=name,
                overwrite_output_path=ctx.overwrite,
                preserve_output_path_name=True,
                create_as_dir=False,
                **spec_tmpl,
            )
        else:
            if not ctx.overwrite:
                logger.info(
                    f"{ctx._manifest_dir.getsyspath(name)} won't be overwritten."
                    " To overwrite pass `--overwrite`",
                    extra={"markup": True},
                )
                return

    @cli.command()
    @pass_environment
    def generate(ctx: Environment) -> None:
        """
        Generate Dockerfile and README for a given docker package.

        \b
        Usage:
            manager generate bento-server --bentoml_version 1.0.0a6
            manager generate bento-server --bentoml_version 1.0.0a6 --dump-metadata

        \b
        NOTE: --dump_metadata is useful when development to see build and releases context.

        """

        xx_info = (ctx.xx_image, ctx.xx_version)
        try:
            if ctx.overwrite:
                ctx._generated_dir.removetree(".")

            # generate readmes and dockerfiles
            generate_dockerfiles(ctx, xx_info)
            generate_readmes(ctx)
            if ctx.verbose:
                logger.info(
                    f"[green] Finished generating {ctx.docker_package}...[/]",
                    extra={"markup": True},
                )
        finally:
            if ctx.verbose:
                logger.debug(
                    "[bold yellow]Dump context to [bold]generated[/]... [/]",
                    extra={"markup": True},
                )

            with ctx._generated_dir.open(
                "generate.meta.json", "w"
            ) as ouf1, ctx._generated_dir.open("tags.meta.json", "w") as ouf2:
                ouf1.write(json.dumps(unstructure(ctx.build_ctx), indent=2))
                ouf2.write(json.dumps(unstructure(ctx.release_ctx), indent=2))


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


def generate_dockerfiles(
    ctx: Environment, xx_info: t.Tuple[t.Optional[str], t.Optional[str]]
):

    xx_image, xx_version = xx_info

    if xx_image and xx_image != ctx.xx_image:
        xx_image_ = xx_image
    else:
        xx_image_ = ctx.xx_image
    if xx_version and xx_image == "local-xx":
        xx_version_ = xx_version
    else:
        xx_version_ = ctx.xx_version

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
                            xx_version=xx_version_,
                            xx_image=xx_image_,
                            metadata=metadata,
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
                        except Exception:  # pylint: disable=broad-except
                            logger.error(traceback.format_exc())
