from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import click

from ._internal._gen import gen_readmes
from ._internal._gen import gen_manifest
from ._internal._gen import gen_dockerfiles
from ._internal._funcs import send_log
from ._internal.groups import pass_environment

if TYPE_CHECKING:
    from ._internal.groups import Environment

logger = logging.getLogger(__name__)


def add_generation_command(cli: click.Group) -> None:
    @cli.command(name="create-manifest")
    @pass_environment
    def create_manifest(ctx: Environment) -> None:  # dead: ignore
        """
        Generate a manifest files to edit.
        Note that we still need to customize this manifest files to fit with our usecase.
        """
        gen_manifest(
            ctx.docker_package,
            ctx.cuda_version,
            ctx.distros,
            overwrite=ctx.overwrite,
            docker_fs=ctx._fs,
        )

    @cli.command()
    @pass_environment
    def generate(ctx: Environment) -> None:  # dead: ignore
        """
        Generate Dockerfile and README for a given docker package.

        \b
        Usage:
            manager generate bento-server --bentoml_version 1.0.0a6
        """
        if ctx.overwrite:
            ctx._generated_dir.removetree("/")

        # generate readmes and dockerfiles
        gen_dockerfiles(ctx)

        gen_readmes(ctx)
        send_log(
            f"[green]Finished generating {ctx.docker_package}...[/]",
            extra={"markup": True},
        )
