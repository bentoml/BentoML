import sys
import json
import logging
from typing import TYPE_CHECKING

import yaml
import click
from simple_di import inject
from simple_di import Provide
from rich.table import Table
from rich.console import Console

from bentoml.bentos import import_bento
from bentoml.bentos import build_bentofile

from ..utils import calc_dir_size
from ..utils import human_readable_size
from .click_utils import _is_valid_bento_tag
from .click_utils import _is_valid_bento_name
from ..yatai_client import yatai_client
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..bento import BentoStore

logger = logging.getLogger(__name__)


def parse_delete_targets_argument_callback(
    ctx, params, value
):  # pylint: disable=unused-argument
    if value is None:
        return value
    delete_targets = value.split(",")
    delete_targets = list(map(str.strip, delete_targets))
    for delete_target in delete_targets:
        if not (
            _is_valid_bento_tag(delete_target) or _is_valid_bento_name(delete_target)
        ):
            raise click.BadParameter(
                "Bad formatting. Please present a valid bento bundle name or "
                '"name:version" tag. For list of bento bundles, separate delete '
                'targets by ",", for example: "my_service:v1,my_service:v2,'
                'classifier"'
            )
    return delete_targets


@inject
def add_bento_management_commands(
    cli,
    bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
):
    @cli.command(help="Get Bento information")
    @click.argument("bento_tag", type=click.STRING)
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["json", "yaml", "path"]),
        default="yaml",
    )
    def get(bento_tag, output):
        """Print Bento details by providing the bento_tag

        bentoml get FraudDetector:latest
        bentoml get --output=json FraudDetector:20210709_DE14C9
        """
        bento = bento_store.get(bento_tag)
        console = Console()

        if output == "path":
            console.print(bento.path)
        elif output == "json":
            info = json.dumps(bento.info.to_dict(), indent=2, default=str)
            console.print_json(info)
        else:
            info = yaml.dump(bento.info, indent=2)
            console.print(info)

    @cli.command(name="list", help="List Bentos in local bento store")
    @click.argument("bento_name", type=click.STRING, required=False)
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["json", "yaml", "table"]),
        default="table",
    )
    @click.option(
        "--no-trunc",
        is_flag=False,
        help="Don't truncate the output",
    )
    def list_bentos(bento_name, output, no_trunc):
        """Print list of bentos in local store

        # show all bentos saved
        > bentoml list

        # show all verions of bento with the name FraudDetector
        > bentoml list FraudDetector
        """
        bentos = bento_store.list(bento_name)
        res = [
            {
                "tag": str(bento.tag),
                "service": bento.info.service,
                "path": bento.path,
                "size": human_readable_size(calc_dir_size(bento.path)),
                "creation_time": bento.info.creation_time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for bento in sorted(
                bentos, key=lambda x: x.info.creation_time, reverse=True
            )
        ]
        if output == "json":
            info = json.dumps(res, indent=2)
            print(info)
        elif output == "yaml":
            info = yaml.safe_dump(res, indent=2)
            print(info)
        else:
            table = Table(box=None)
            table.add_column("Tag")
            table.add_column("Service")
            table.add_column("Path")
            table.add_column("Size")
            table.add_column("Creation Time")
            for bento in res:
                table.add_row(
                    bento["tag"],
                    bento["service"],
                    bento["path"],
                    bento["size"],
                    bento["creation_time"],
                )
            console = Console()
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
    def delete(
        delete_targets,
        yes,
    ):
        """Delete Bento in local bento store.

        Specify target Bentos to remove:

        * Delete single bento bundle by "name:version", e.g: `bentoml delete IrisClassifier:v1`
        * Bulk delete all bento bundles with a specific name, e.g.: `bentoml delete IrisClassifier`
        * Bulk delete multiple bento bundles by name and version, separated by ",", e.g.: `benotml delete Irisclassifier:v1,MyPredictService:v2`
        """  # noqa

        def delete_target(target):
            to_delete_bentos = bento_store.list(target)

            for bento in to_delete_bentos:
                if yes:
                    delete_confirmed = True
                else:
                    delete_confirmed = click.confirm(f"delete bento {bento.tag}?")

                if delete_confirmed:
                    bento_store.delete(bento.tag)
                    logger.info(f"{bento} deleted")

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
    def export(bento_tag, out_path):
        """Export Bento files to an archive file

        \b
        arguments:
        BENTO_TAG: bento identifier
        OUT_PATH: output path of exported bento.
          If this argument is not provided, bento is exported to name-version.bento in the current directory.
          Supported formats are tar ('tar'), tar.gz ('gz'), tar.xz ('xz'), tar.bz2 ('bz2'), and zip

        \b
        examples:
        bentoml export FraudDetector:20210709_DE14C9
        bentoml export FraudDetector:20210709_DE14C9 ./my_bento.tar
        bentoml export FraudDetector:latest ./my_bento.zip
        bentoml export FraudDetector:latest s3://mybucket/bentos/my_bento.gz
        """
        bento = bento_store.get(bento_tag)
        bento.export(out_path)
        logger.info(f"{bento} exported to {out_path}")

    @cli.command(name="import")
    @click.argument("bento_path", type=click.STRING)
    def import_from(bento_path):
        """Import a previously exported Bento archive file

        \b
        argument:
        BENTO_PATH: path of Bento archive file

        \b
        examples:
        bentoml import ./my_bento.tar
        bentoml import s3://mybucket/bentos/my_bento.zip
        """
        bento = import_bento(bento_path)
        logger.info(f"{bento} imported")

    @cli.command(
        help="Pull Bento from a yatai server",
    )
    @click.argument("bento_tag", type=click.STRING)
    @click.option(
        "-f",
        "--force",
        is_flag=True,
        default=False,
        help="Force pull from yatai to local and overwrite even if it already exists in local",
    )
    def pull(bento_tag: str, force: bool):
        yatai_client.pull_bento(bento_tag, force=force)

    @cli.command(help="Push Bento to a yatai server")
    @click.argument("bento_tag", type=click.STRING)
    @click.option(
        "-f",
        "--force",
        is_flag=True,
        default=False,
        help="Forced push to yatai even if it exists in yatai",
    )
    def push(bento_tag: str, force: bool):
        bento_obj = bento_store.get(bento_tag)
        if not bento_obj:
            raise click.ClickException(f"Bento {bento_tag} not found in local store")
        yatai_client.push_bento(bento_obj, force=force)

    @cli.command(help="Build a new Bento from current directory")
    @click.argument("build_ctx", type=click.Path(), default=".")
    @click.option("-f", "--bentofile", type=click.STRING, default="bentofile.yaml")
    @click.option("--version", type=click.STRING, default=None)
    def build(build_ctx, bentofile, version):
        if sys.path[0] != build_ctx:
            sys.path.insert(0, build_ctx)

        build_bentofile(bentofile, build_ctx=build_ctx, version=version)
