from __future__ import annotations

import json
import typing as t

import yaml
import click
from simple_di import inject
from simple_di import Provide
from rich.table import Table
from rich.syntax import Syntax

from bentoml_cli.utils import is_valid_bento_tag
from bentoml_cli.utils import is_valid_bento_name

if t.TYPE_CHECKING:
    from click import Group
    from click import Context
    from click import Parameter

    from bentoml._internal.bento import BentoStore

BENTOML_FIGLET = """
██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝
"""


def parse_delete_targets_argument_callback(
    ctx: Context, params: Parameter, value: t.Any  # pylint: disable=unused-argument
) -> list[str]:
    if value is None:
        return value
    value = " ".join(value)
    if "," in value:
        delete_targets = value.split(",")
    else:
        delete_targets = value.split()
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
    import bentoml
    from bentoml import Tag
    from bentoml.bentos import import_bento
    from bentoml._internal.utils import rich_console as console
    from bentoml._internal.utils import calc_dir_size
    from bentoml._internal.utils import human_readable_size
    from bentoml._internal.utils import resolve_user_filepath
    from bentoml._internal.bento.bento import Bento
    from bentoml._internal.bento.bento import DEFAULT_BENTO_BUILD_FILE
    from bentoml._internal.configuration import get_quiet_mode
    from bentoml._internal.bento.build_config import BentoBuildConfig
    from bentoml._internal.configuration.containers import BentoMLContainer
    from bentoml._internal.utils.analytics.usage_stats import _usage_event_debugging

    bento_store = BentoMLContainer.bento_store.get()
    yatai_client = BentoMLContainer.yatai_client.get()

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
            console.print(Syntax(info, "yaml", background_color="default"))

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
            console.print(Syntax(info, "yaml", background_color="default"))
        else:
            table = Table(box=None)
            table.add_column("Tag")
            table.add_column("Size")
            table.add_column("Creation Time")
            for bento in res:
                table.add_row(
                    bento["tag"],
                    bento["size"],
                    bento["creation_time"],
                )
            console.print(table)

    @cli.command()
    @click.argument(
        "delete_targets",
        nargs=-1,
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
            * Bulk delete multiple bento bundles by name and version, separated by ",", e.g.: `bentoml delete Irisclassifier:v1,MyPredictService:v2`
            * Bulk delete multiple bento bundles by name and version, separated by " ", e.g.: `bentoml delete Irisclassifier:v1 MyPredictService:v2`
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
                    click.echo(f"{bento} deleted.")

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
        click.echo(f"{bento} exported to {out_path}.")

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
        click.echo(f"{bento} imported.")

    @cli.command()
    @click.argument("bento_tag", type=click.STRING)
    @click.option(
        "-f",
        "--force",
        is_flag=True,
        default=False,
        help="Force pull from yatai to local and overwrite even if it already exists in local",
    )
    @click.option(
        "--context", type=click.STRING, default=None, help="Yatai context name."
    )
    def pull(bento_tag: str, force: bool, context: str) -> None:  # type: ignore (not accessed)
        """Pull Bento from a yatai server."""
        yatai_client.pull_bento(bento_tag, force=force, context=context)

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
    @click.option(
        "--context", type=click.STRING, default=None, help="Yatai context name."
    )
    def push(bento_tag: str, force: bool, threads: int, context: str) -> None:  # type: ignore (not accessed)
        """Push Bento to a yatai server."""
        bento_obj = bento_store.get(bento_tag)
        if not bento_obj:
            raise click.ClickException(f"Bento {bento_tag} not found in local store")
        yatai_client.push_bento(
            bento_obj, force=force, threads=threads, context=context
        )

    @cli.command()
    @click.argument("build_ctx", type=click.Path(), default=".")
    @click.option(
        "-f",
        "--bentofile",
        type=click.STRING,
        default=DEFAULT_BENTO_BUILD_FILE,
        help="Path to bentofile. Default to 'bentofile.yaml'",
    )
    @click.option(
        "--version",
        type=click.STRING,
        default=None,
        help="Bento version. By default the version will be generated.",
    )
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["tag", "default"]),
        default="default",
        show_default=True,
        help="Output log format. '-o tag' to display only bento tag.",
    )
    @inject
    def build(  # type: ignore (not accessed)
        build_ctx: str,
        bentofile: str,
        version: str,
        output: t.Literal["tag", "default", "yaml"],
        _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    ):
        """Build a new Bento from current directory."""
        if output == 'tag':
            from bentoml._internal.configuration import set_quiet_mode

            from bentoml._internal.log import configure_logging

            set_quiet_mode(True)
            configure_logging()

        try:
            bentofile = resolve_user_filepath(bentofile, build_ctx)
        except FileNotFoundError:
            raise bentoml.exceptions.InvalidArgument(
                f'bentofile "{bentofile}" not found'
            )

        with open(bentofile, "r", encoding="utf-8") as f:
            build_config = BentoBuildConfig.from_yaml(f)

        bento = Bento.create(
            build_config=build_config,
            version=version,
            build_ctx=build_ctx,
        ).save(_bento_store)

        # NOTE: Don't remove the return statement here, since we will need this
        # for usage stats collection if users are opt-in.
        if output == "tag":
            if _usage_event_debugging():
                # NOTE: Since we are logging all of the trackintg id to stdout
                # We will prefix the tag with __tag__ and we can use regex to correctly
                # get the tag from 'bentoml.bentos.build|build_bentofile'
                click.echo(f"__tag__:{bento.tag}")
            else:
                # Current behaviour
                click.echo(bento.tag)
            return bento

        if not get_quiet_mode():
            click.echo(BENTOML_FIGLET)
            click.secho(f"Successfully built {bento}.", fg="green")

            click.secho(
                f"\nPossible next steps:\n\n * Containerize your Bento with `bentoml containerize`:\n    $ bentoml containerize {bento.tag}",
                fg="blue",
            )
            click.secho(
                f"\n * Push to BentoCloud with `bentoml push`:\n    $ bentoml push {bento.tag}",
                fg="blue",
            )
        return bento
