from __future__ import annotations

import json
import os
import typing as t
from http import HTTPStatus

import click
import rich
import yaml
from rich.syntax import Syntax
from rich.table import Table
from simple_di import Provide
from simple_di import inject

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.utils.filesystem import resolve_user_filepath
from bentoml.exceptions import BentoMLException
from bentoml_cli.utils import BentoMLCommandGroup

if t.TYPE_CHECKING:
    from click import Context
    from click import Parameter

    from bentoml._internal.cloud import BentoCloudClient


@click.group(name="secret", cls=BentoMLCommandGroup)
def secret_command():
    """Secret Subcommands Groups"""


@secret_command.command(name="list")
@click.option(
    "--search", type=click.STRING, default=None, help="Search for list request."
)
@click.option(
    "-o",
    "--output",
    help="Display the output of this command.",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
)
@inject
def list_command(
    search: str | None,
    output: t.Literal["json", "yaml", "table"],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
):
    """List all secrets on BentoCloud."""
    try:
        secrets = _cloud_client.secret.list(search=search)
    except BentoMLException as e:
        raise_secret_error(e, "list")
    if output == "table":
        table = Table(box=None, expand=True)
        table.add_column("Secret", overflow="fold")
        table.add_column("Created_At", overflow="fold")
        table.add_column("Mount_As", overflow="fold")
        table.add_column("Keys", overflow="fold")
        table.add_column("Path", overflow="fold")
        table.add_column("Cluster", overflow="fold")

        for secret in secrets:
            keys = [item.key for item in secret.content.items]
            mountAs = secret.content.type
            if mountAs == "env":
                mountAs = "Environment Variable"
            elif mountAs == "mountfile":
                mountAs = "File"
            table.add_row(
                secret.name,
                secret.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                mountAs,
                ", ".join(keys),
                secret.content.path if secret.content.path else "-",
                secret.cluster.name,
            )
        rich.print(table)
    elif output == "json":
        res: t.List[dict[str, t.Any]] = [s.to_dict() for s in secrets]
        info = json.dumps(res, indent=2, default=str)
        rich.print(info)
    elif output == "yaml":
        res: t.List[dict[str, t.Any]] = [s.to_dict() for s in secrets]
        info = yaml.dump(res, indent=2, sort_keys=False)
        rich.print(Syntax(info, "yaml", background_color="default"))


def parse_kvs_argument_callback(
    ctx: Context,
    params: Parameter,
    value: tuple[str, ...],  # pylint: disable=unused-argument
) -> t.List[tuple[str, str]]:
    """
    split "key1=value1 key2=value2" into [("key1", "value1"), ("key2", "value2")],
    """
    key_vals: t.List[tuple[str, str]] = []
    for key_val in value:
        key, val = key_val.split("=")
        if not key or not val:
            raise click.BadParameter(f"Invalid key-value pair: {key_val}")
        if val.startswith("@"):
            filename = resolve_user_filepath(val[1:], ctx=None)
            if not os.path.exists(filename) or not os.path.isfile(filename):
                raise click.BadParameter(f"Invalid file path: {filename}")
            # read the file content
            with open(filename, "r") as f:
                val = f.read()
        key_vals.append((key, val))
    return key_vals


def read_dotenv_callback(
    ctx: Context,
    params: Parameter,
    value: tuple[str, ...],  # pylint: disable=unused-argument
) -> t.List[tuple[str, str]]:
    from dotenv import dotenv_values

    env_map: dict[str, str] = {}

    for path in value:
        path = resolve_user_filepath(path, ctx=None)
        if not os.path.exists(path) or not os.path.isfile(path):
            raise click.BadParameter(f"Invalid file path: {path}")
        values = {k: v for k, v in dotenv_values(path).items() if v is not None}
        env_map.update(values)
    return list(env_map.items())


def raise_secret_error(err: BentoMLException, action: str) -> t.NoReturn:
    if err.error_code == HTTPStatus.UNAUTHORIZED:
        raise BentoMLException(
            f"{err}\n* BentoCloud API token is required for authorization. Run `bentoml cloud login` command to login"
        ) from None
    elif err.error_code == HTTPStatus.NOT_FOUND:
        raise BentoMLException("Secret not found") from None
    raise BentoMLException(f"Failed to {action} secret due to: {err}")


def map_choice_to_type(ctx: Context, params: Parameter, value: t.Any):
    mappings = {"env": "env", "file": "mountfile"}
    return mappings[value]


@secret_command.command(name="create")
@click.argument(
    "name",
    nargs=1,
    type=click.STRING,
    required=True,
)
@click.argument(
    "key_vals",
    nargs=-1,
    type=click.STRING,
    callback=parse_kvs_argument_callback,
)
@click.option(
    "-d",
    "--description",
    type=click.STRING,
    help="Secret description",
)
@click.option(
    "-t",
    "--type",
    type=click.Choice(["env", "file"]),
    help="Mount as Environment Variable or File",
    default="env",
    callback=map_choice_to_type,
)
@click.option(
    "--cluster",
    type=click.STRING,
    help="Name of the cluster",
)
@click.option(
    "-p",
    "--path",
    type=click.STRING,
    help="Path where the secret will be mounted in the container. The path must be under the ($BENTOML_HOME) directory.",
)
@click.option(
    "-l",
    "--from-literal",
    help="Pass key value pairs by --from-literal key1=value1 key2=value2",
    is_flag=True,
    hidden=True,
)
@click.option(
    "-f",
    "--from-file",
    metavar="DOTENV_FILE",
    help="Read environment variables from dotenv file",
    callback=read_dotenv_callback,
    multiple=True,
)
@inject
def create(
    name: str,
    description: str | None,
    type: t.Literal["env", "mountfile"],
    cluster: str | None,
    path: str | None,
    key_vals: t.List[tuple[str, str]],
    from_literal: bool,
    from_file: t.List[tuple[str, str]],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
):
    """Create a secret on BentoCloud.

    Pass key value pairs by key1=value1 key2=value2

    Pass key value from file by key1=@./path_to_file1 key2=@./path_to_file2
    """
    try:
        if from_literal:
            click.echo(
                "--from-literal is deprecated and does not take effect.", err=True
            )

        key_vals.extend(from_file)

        if not key_vals:
            raise click.BadParameter("At least one key-value pair is required")

        if type == "mountfile" and not path:
            path = "$BENTOML_HOME"
        secret = _cloud_client.secret.create(
            name=name,
            type=type,
            cluster=cluster,
            description=description,
            path=path,
            key_vals=key_vals,
        )
        rich.print(f"Secret [green]{secret.name}[/] created successfully")
    except BentoMLException as e:
        raise_secret_error(e, "create")


@secret_command.command(name="delete")
@click.argument(
    "name",
    nargs=1,
    type=click.STRING,
    required=True,
)
@click.option(
    "--cluster",
    type=click.STRING,
    help="Name of the cluster",
)
@inject
def delete(
    name: str,
    cluster: str | None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
):
    """Delete a secret on BentoCloud."""
    try:
        _cloud_client.secret.delete(name=name, cluster=cluster)
        rich.print(f"Secret [green]{name}[/] deleted successfully")
    except BentoMLException as e:
        raise_secret_error(e, "delete")


@secret_command.command(name="apply")
@click.argument(
    "name",
    nargs=1,
    type=click.STRING,
    required=True,
)
@click.option(
    "--cluster",
    type=click.STRING,
    help="Name of the cluster",
)
@click.argument(
    "key_vals",
    nargs=-1,
    type=click.STRING,
    callback=parse_kvs_argument_callback,
)
@click.option(
    "-d",
    "--description",
    type=click.STRING,
    help="Secret description",
)
@click.option(
    "-t",
    "--type",
    type=click.Choice(["env", "file"]),
    help="Mount as Environment Variable or File",
    default="env",
    callback=map_choice_to_type,
)
@click.option(
    "-p",
    "--path",
    type=click.STRING,
    help="Path where the secret will be mounted in the container. The path must be under the ($BENTOML_HOME) directory.",
)
@click.option(
    "-l",
    "--from-literal",
    help="Pass key value pairs by --from-literal key1=value1 key2=value2",
    hidden=True,
    is_flag=True,
)
@click.option(
    "-f",
    "--from-file",
    metavar="DOTENV_FILE",
    help="Read environment variables from dotenv file",
    callback=read_dotenv_callback,
    multiple=True,
)
@inject
def apply(
    name: str,
    description: str | None,
    cluster: str | None,
    type: t.Literal["env", "mountfile"],
    path: str | None,
    key_vals: t.List[t.Tuple[str, str]],
    from_literal: bool,
    from_file: t.List[t.Tuple[str, str]],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
):
    """Apply a secret update on BentoCloud.

    Pass key value pairs by key1=value1 key2=value2

    Pass key value from file by key1=@./path_to_file1 key2=@./path_to_file2
    """
    try:
        if from_literal:
            click.echo(
                "--from-literal is deprecated and does not take effect.", err=True
            )

        key_vals.extend(from_file)

        if not key_vals:
            raise click.BadParameter("At least one key-value pair is required")

        if type == "mountfile" and not path:
            path = "$BENTOML_HOME"
        secret = _cloud_client.secret.update(
            name=name,
            type=type,
            cluster=cluster,
            description=description,
            path=path,
            key_vals=key_vals,
        )
        rich.print(f"Secret [green]{secret.name}[/] applied successfully")
    except BentoMLException as e:
        raise_secret_error(e, "apply")
