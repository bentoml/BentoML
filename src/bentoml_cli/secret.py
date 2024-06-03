from __future__ import annotations
import os
import json
import click
import typing as t
from http import HTTPStatus

import yaml
from rich.syntax import Syntax
from rich.table import Table

from bentoml_cli.utils import BentoMLCommandGroup
from bentoml._internal.cloud.secret import Secret
from bentoml._internal.utils import rich_console as console
from bentoml.exceptions import BentoMLException

if t.TYPE_CHECKING:
    from click import Context
    from click import Parameter

if t.TYPE_CHECKING:
    TupleStrAny = tuple[str, ...]
    from .utils import SharedOptions
else:
    TupleStrAny = tuple


@click.group(name="secret", cls=BentoMLCommandGroup)
def secret_command():
    """Secret Subcommands Groups"""
    
    
@secret_command.command(name="list")
@click.option(
    "--search", 
    type=click.STRING,
    default=None,
    help="Search for list request."
)
@click.pass_obj
@click.option(
    "-o",
    "--output",
    help="Display the output of this command.",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
)
def list(
    shared_options: SharedOptions,
    search: str | None,
    output: t.Literal["json", "yaml", "table"],
):
    """List all secrets"""
    secrets = Secret.list(context=shared_options.cloud_context, search=search)
    if output == "table":
        table = Table(box=None, expand=True)
        table.add_column("Secret", overflow="fold")
        table.add_column("Created_At", overflow="fold")
        table.add_column("Mount_As", overflow="fold")
        table.add_column("Keys", overflow="fold")
        table.add_column("Path", overflow="fold")
        
        for secret in secrets:
            keys = [item.key for item in secret.content.items]
            table.add_row(
                secret.name,
                secret.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                secret.content.type,
                ", ".join(keys),
                secret.content.path if secret.content.path else "-",
            )
        console.print(table)
    elif output == "json":
        res: t.List[dict[str, t.Any]] = [s.to_dict() for s in secrets]
        info = json.dumps(res, indent=2, default=str)
        console.print(info)
    elif output == "yaml":
        res: t.List[dict[str, t.Any]] = [s.to_dict() for s in secrets]
        info = yaml.dump(res,  indent=2, sort_keys=False)
        console.print(Syntax(info, "yaml", background_color="default"))


def parse_kvs_argument_callback(
    ctx: Context,
    params: Parameter,
    value: t.Any,  # pylint: disable=unused-argument
) -> t.List[t.Tuple[str, str]]:
    '''
    split "key1=value1 key2=value2" into [("key1", "value1"), ("key2", "value2")],
    '''
    key_vals : t.List[t.Tuple[str, str]] = []
    for key_val in value:
        key, val = key_val.split("=")
        if not key or not val:
            raise click.BadParameter(f"Invalid key-value pair: {key_val}")
        key_vals.append((key, val))
    return key_vals

def parse_from_literal_argument_callback(
    ctx: Context,
    params: Parameter,
    value: t.Any,  # pylint: disable=unused-argument
) -> t.List[t.Tuple[str, str]]:
    '''
    split "key1=value1 key2=value2" into [("key1", "value1"), ("key2", "value2")],
    '''
    from_literal : t.List[t.Tuple[str, str]] = []
    for key_val in value:
        key, val = key_val.split("=")
        if not key or not val:
            raise click.BadParameter(f"Invalid key-value pair: {key_val}")
        from_literal.append((key, val))
    return from_literal

def parse_from_file_argument_callback(
    ctx: Context,
    params: Parameter,
    value: t.Any,  # pylint: disable=unused-argument
) -> t.List[t.Tuple[str, str]]:
    '''
    split "key1=value1 key2=value2" into [("key1", "value1"), ("key2", "value2")],
    '''
    from_file : t.List[t.Tuple[str, str]] = []
    for key_path in value:
        key, path = key_path.split("=")
        if not key or not path:
            raise click.BadParameter(f"Invalid key-path pair: {key_path}")
        if not os.path.exists(path) or not os.path.isfile(path):
            raise click.BadParameter(f"Invalid path: {path}")
        # read the file content
        with open(path, "r") as f:
            val = f.read()
        from_file.append((key, val))
    return from_file

def raise_secret_error(err: BentoMLException, action: str) -> t.NoReturn:
    if err.error_code == HTTPStatus.UNAUTHORIZED:
        raise BentoMLException(
            f"{err}\n* BentoCloud sign up: https://cloud.bentoml.com/\n"
            "* Login with your API token: "
            "https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html"
        ) from None
    raise BentoMLException(
        f"Failed to {action} secret due to invalid configuration: {err}"
    ) from None


@secret_command.command(name="create")
@click.pass_obj
@click.option(
    "-n",
    "--name",
    type=click.STRING,
    help="Secret name",
    required=True,
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
)
@click.option(
    "-p",
    "--path",
    type=click.STRING,
    help="Path where the secret will be mounted in the container",
    default="$BENTOML_HOME",
)
@click.argument(
    "key_vals",
    nargs=-1,
    type=click.STRING,
    callback=parse_kvs_argument_callback,
)
@click.option(
    "-l",
    "--from-literal",
    type=click.STRING,
    help="Pass key value pairs by --from-literal key1=value1 key2=value2",
    callback=parse_from_literal_argument_callback,
    multiple=True,
)
@click.option(
    "-f",
    "--from-file",
    type=click.STRING,
    help="Pass key value pairs by --from-file key1=./path_to_file1 key2=./path_to_file2",
    callback=parse_from_file_argument_callback,
    multiple=True,
)
def create(
    shared_options: SharedOptions,
    name: str, 
    description: str | None, 
    type: t.Literal["env", "file"], 
    path: str | None,
    key_vals: t.List[t.Tuple[str, str]],
    from_literal: t.List[t.Tuple[str, str]],
    from_file: t.List[t.Tuple[str, str]],
):
    """Create a secret"""
    try:
        if from_literal and from_file:
            raise BentoMLException("options --from-literal and --from-file can not be used together")
        
        key_vals.extend(from_literal)
        key_vals.extend(from_file)
        
        if not key_vals:
            raise BentoMLException("no key-value pairs provided, please use --from-literal or --from-file or provide key-value pairs")
        
        secret = Secret.create(context=shared_options.cloud_context, name=name, description=description, type=type, path=path, key_vals=key_vals)
        click.echo(f"Secret {secret.name} created successfully")
    except Exception as e:
        raise_secret_error(e, "create")
        