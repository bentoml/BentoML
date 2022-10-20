from __future__ import annotations

import sys
import json
import typing as t
import logging
from typing import TYPE_CHECKING

import yaml
import click
from rich.table import Table
from rich.syntax import Syntax

from bentoml_cli.utils import is_valid_bento_tag
from bentoml_cli.utils import is_valid_bento_name

if TYPE_CHECKING:
    from click import Group
    from click import Context
    from click import Parameter

logger = logging.getLogger("bentoml")


def parse_delete_targets_argument_callback(
    ctx: Context, params: Parameter, value: t.Any  # pylint: disable=unused-argument
) -> list[str]:
    if value is None:
        return value
    delete_targets = value.split(",")
    delete_targets = list(map(str.strip, delete_targets))
    for delete_target in delete_targets:
        if not (
            is_valid_bento_tag(delete_target) or is_valid_bento_name(delete_target)
        ):
            raise click.BadParameter(
                f'Bad formatting: "{delete_target}". Please present a valid bento bundle name or "name:version" tag. For list of bento bundles, separate delete targets by ",", for example: "my_service:v1,my_service:v2,classifier"'
            )
    return delete_targets


def add_bento_management_commands(cli: Group):
    from bentoml import Tag
    from bentoml.bentos import import_bento
    from bentoml.bentos import build_bentofile
    from bentoml._internal.utils import rich_console as console
    from bentoml._internal.utils import calc_dir_size
    from bentoml._internal.utils import human_readable_size
    from bentoml._internal.utils import display_path_under_home
    from bentoml._internal.bento.bento import DEFAULT_BENTO_BUILD_FILE
    from bentoml._internal.yatai_client import yatai_client
    from bentoml._internal.configuration.containers import BentoMLContainer

    bento_store = BentoMLContainer.bento_store.get()

    @cli.command()
    @click.argument("bento_tag", type=click.STRING)
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["json", "yaml", "path"]),
        default="yaml",
    )
    def get(bento_tag: str, output: str) -> None:  # type: ignore (not accessed)
        """Print Bento details by providing the bento_tag.

        \b
        bentoml get iris_classifier:qojf5xauugwqtgxi
        bentoml get iris_classifier:qojf5xauugwqtgxi --output=json
        """
        bento = bento_store.get(bento_tag)

        if output == "path":
            console.print(bento.path)
        elif output == "json":
            info = json.dumps(bento.info.to_dict(), indent=2, default=str)
            console.print_json(info)
        else:
            info = yaml.dump(bento.info, indent=2, sort_keys=False)
            console.print(Syntax(info, "yaml"))

    @cli.command(name="list")
    @click.argument("bento_name", type=click.STRING, required=False)
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["json", "yaml", "table"]),
        default="table",
    )
    def list_bentos(bento_name: str, output: str) -> None:  # type: ignore (not accessed)
        """List Bentos in local store

        \b
        # show all bentos saved
        $ bentoml list

        \b
        # show all verions of bento with the name FraudDetector
        $ bentoml list FraudDetector
        """
        bentos = bento_store.list(bento_name)
        res = [
            {
                "tag": str(bento.tag),
                "path": display_path_under_home(bento.path),
                "size": human_readable_size(calc_dir_size(bento.path)),
                "creation_time": bento.info.creation_time.astimezone().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
            for bento in sorted(
                bentos, key=lambda x: x.info.creation_time, reverse=True
            )
        ]

        if output == "json":
            info = json.dumps(res, indent=2)
            console.print(info)
        elif output == "yaml":
            info = yaml.safe_dump(res, indent=2)
            console.print(Syntax(info, "yaml"))
        else:
            table = Table(box=None)
            table.add_column("Tag")
            table.add_column("Size")
            table.add_column("Creation Time")
            table.add_column("Path")
            for bento in res:
                table.add_row(
                    bento["tag"],
                    bento["size"],
                    bento["creation_time"],
                    bento["path"],
                )
            console.print(table)

    @cli.command()
    @click.argument(
        "delete_targets",
        type=click.STRING,
        callback=parse_delete_targets_argument_callback,
        required=True,
    )
    @click.option(
        "-y",
        "--yes",
        "--assume-yes",
        is_flag=True,
        help="Skip confirmation when deleting a specific bento bundle",
    )
    def delete(delete_targets: list[str], yes: bool) -> None:  # type: ignore (not accessed)
        """Delete Bento in local bento store.

        \b
        Examples:
            * Delete single bento bundle by "name:version", e.g: `bentoml delete IrisClassifier:v1`
            * Bulk delete all bento bundles with a specific name, e.g.: `bentoml delete IrisClassifier`
            * Bulk delete multiple bento bundles by name and version, separated by ",", e.g.: `benotml delete Irisclassifier:v1,MyPredictService:v2`
            * Bulk delete without confirmation, e.g.: `bentoml delete IrisClassifier --yes`
        """

        def delete_target(target: str) -> None:
            tag = Tag.from_str(target)

            if tag.version is None:
                to_delete_bentos = bento_store.list(target)
            else:
                to_delete_bentos = [bento_store.get(tag)]

            for bento in to_delete_bentos:
                if yes:
                    delete_confirmed = True
                else:
                    delete_confirmed = click.confirm(f"delete bento {bento.tag}?")

                if delete_confirmed:
                    bento_store.delete(bento.tag)
                    logger.info("%s deleted.", bento)

        for target in delete_targets:
            delete_target(target)

    @cli.command()
    @click.argument("bento_tag", type=click.STRING)
    @click.argument(
        "out_path",
        type=click.STRING,
        default="",
        required=False,
    )
    def export(bento_tag: str, out_path: str) -> None:  # type: ignore (not accessed)
        """Export a Bento to an external file archive

        \b
        Arguments:
            BENTO_TAG: bento identifier
            OUT_PATH: output path of exported bento.

        If out_path argument is not provided, bento is exported to name-version.bento in the current directory.
        Beside the native .bento format, we also support ('tar'), tar.gz ('gz'), tar.xz ('xz'), tar.bz2 ('bz2'), and zip.

        \b
        Examples:
            bentoml export FraudDetector:20210709_DE14C9
            bentoml export FraudDetector:20210709_DE14C9 ./my_bento.bento
            bentoml export FraudDetector:latest ./my_bento.bento
            bentoml export FraudDetector:latest s3://mybucket/bentos/my_bento.bento
        """
        bento = bento_store.get(bento_tag)
        out_path = bento.export(out_path)
        logger.info("%s exported to %s.", bento, out_path)

    @cli.command(name="import")
    @click.argument("bento_path", type=click.STRING)
    def import_bento_(bento_path: str) -> None:  # type: ignore (not accessed)
        """Import a previously exported Bento archive file

        \b
        Arguments:
            BENTO_PATH: path of Bento archive file

        \b
        Examples:
            bentoml import ./my_bento.bento
            bentoml import s3://mybucket/bentos/my_bento.bento
        """
        bento = import_bento(bento_path)
        logger.info("%s imported.", bento)

    @cli.command()
    @click.argument("bento_tag", type=click.STRING)
    @click.option(
        "-f",
        "--force",
        is_flag=True,
        default=False,
        help="Force pull from yatai to local and overwrite even if it already exists in local",
    )
    def pull(bento_tag: str, force: bool) -> None:  # type: ignore (not accessed)
        """Pull Bento from a yatai server."""
        yatai_client.pull_bento(bento_tag, force=force)

    @cli.command()
    @click.argument("bento_tag", type=click.STRING)
    @click.option(
        "-f",
        "--force",
        is_flag=True,
        default=False,
        help="Forced push to yatai even if it exists in yatai",
    )
    @click.option(
        "-t",
        "--threads",
        default=10,
        help="Number of threads to use for upload",
    )
    def push(bento_tag: str, force: bool, threads: int) -> None:  # type: ignore (not accessed)
        """Push Bento to a yatai server."""
        bento_obj = bento_store.get(bento_tag)
        if not bento_obj:
            raise click.ClickException(f"Bento {bento_tag} not found in local store")
        yatai_client.push_bento(bento_obj, force=force, threads=threads)

    @cli.command()
    @click.argument("build_ctx", type=click.Path(), default=".")
    @click.option(
        "-f", "--bentofile", type=click.STRING, default=DEFAULT_BENTO_BUILD_FILE
    )
    @click.option("--version", type=click.STRING, default=None)
    def build(build_ctx: str, bentofile: str, version: str) -> None:  # type: ignore (not accessed)
        """Build a new Bento from current directory."""
        if sys.path[0] != build_ctx:
            sys.path.insert(0, build_ctx)

        build_bentofile(bentofile, build_ctx=build_ctx, version=version)
