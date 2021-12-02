import sys

import click

from .click_utils import _is_valid_bento_tag
from .click_utils import _is_valid_bento_name


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


def add_bento_management_commands(cli):
    @cli.command(help="Get Bento information")
    @click.argument("bento_tag", type=click.STRING)
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["tree", "json", "yaml", "path"]),
        default="tree",
    )
    def get(bento_tag, output):
        """Print Bento details by providing the bento_tag

        bentoml get FraudDetector:latest
        bentoml get FraudDetector:20210709_DE14C9
        """
        pass

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
        pass

    @cli.command()
    @click.argument(
        "delete_targets",
        type=click.STRING,
        callback=parse_delete_targets_argument_callback,
        required=False,
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
        pass

    @cli.command(help="Export Bento to a tar file")
    @click.argument("bento_tag", type=click.STRING)
    @click.argument(
        "out_file", type=click.File("wb"), default=sys.stdout, required=False
    )
    def export(bento_tag, out_file):
        """Export Bento files to a tar file

        bentoml export FraudDetector:latest > my_bento.tar
        bentoml export FraudDetector:20210709_DE14C9 ./my_bento.tar
        """
        pass

    @cli.command(name="import", help="Import a previously exported Bento tar file")
    @click.argument(
        "bento_path", type=click.File("rb"), default=sys.stdin, required=False
    )
    def import_bento(bento_path):
        """Export Bento files to a tar file

        bentoml import < ./my_bento.tar
        bentoml import ./my_bento.tar
        """
        pass

    @cli.command(
        help="Pull Bento from a yatai server",
    )
    @click.argument("bento", type=click.STRING)
    @click.option(
        "--yatai",
        help='Yatai URL or name (when previous configured). Example: "--yatai=http://localhost:50050"',
    )
    def pull(bento, yatai):
        pass

    @cli.command(help="Push Bento to a yatai server")
    @click.argument("bento", type=click.STRING)
    @click.option(
        "--yatai",
        help='Yatai URL or name (when previous configured). Example: "--yatai=http://localhost:50050"',
    )
    def push(bento, yatai):
        pass
