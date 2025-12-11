from __future__ import annotations

import typing as t

import click
import rich

import bentoml.api_token
from bentoml.exceptions import BentoMLException
from bentoml_cli.utils import BentoMLCommandGroup


@click.group(name="api-token", cls=BentoMLCommandGroup)
def api_token_command():
    """API Token management commands."""


@api_token_command.command(name="list")
@click.option(
    "--search", type=click.STRING, default=None, help="Search for tokens by name."
)
@click.option(
    "-o",
    "--output",
    help="Display the output of this command.",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
)
def list_api_tokens(
    search: str | None,
    output: str,
) -> None:
    """List all API tokens on BentoCloud."""
    import json as json_mod

    import yaml
    from rich.syntax import Syntax
    from rich.table import Table

    try:
        tokens = bentoml.api_token.list(search=search)
    except BentoMLException as e:
        _raise_api_token_error(e, "list")

    if output == "table":
        table = Table(box=None, expand=True)
        table.add_column("Name", overflow="fold")
        table.add_column("UID", overflow="fold")
        table.add_column("Created_At", overflow="fold")
        table.add_column("Expired_At", overflow="fold")
        table.add_column("Last_Used_At", overflow="fold")
        table.add_column("Scopes", overflow="fold")

        for token in tokens:
            expired_at = (
                token.expired_at.strftime("%Y-%m-%d %H:%M:%S")
                if token.expired_at
                else "Never"
            )
            last_used_at = (
                token.last_used_at.strftime("%Y-%m-%d %H:%M:%S")
                if token.last_used_at
                else "Never"
            )
            table.add_row(
                token.name,
                token.uid,
                token.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                expired_at,
                last_used_at,
                ", ".join(token.scopes) if token.scopes else "-",
            )
        rich.print(table)
    elif output == "json":
        res: t.List[dict[str, t.Any]] = [t.to_dict() for t in tokens]
        info = json_mod.dumps(res, indent=2, default=str)
        rich.print(info)
    elif output == "yaml":
        res: t.List[dict[str, t.Any]] = [t.to_dict() for t in tokens]
        info = yaml.dump(res, indent=2, sort_keys=False)
        rich.print(Syntax(info, "yaml", background_color="default"))


def _parse_expiration(expires_str: str | None) -> "t.Any":
    """Parse expiration string into datetime."""
    from datetime import datetime
    from datetime import timedelta

    if not expires_str:
        return None

    expires_str = expires_str.strip()

    # Try parsing duration format (e.g., 30d, 1w, 24h)
    if expires_str[-1].lower() in ("d", "w", "h"):
        try:
            value = int(expires_str[:-1])
            unit = expires_str[-1].lower()
            if unit == "d":
                return datetime.now() + timedelta(days=value)
            elif unit == "w":
                return datetime.now() + timedelta(weeks=value)
            elif unit == "h":
                return datetime.now() + timedelta(hours=value)
        except ValueError:
            pass

    # Try parsing ISO date format
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(expires_str, fmt)
        except ValueError:
            continue

    raise click.BadParameter(
        f"Invalid expiration format: {expires_str}. "
        "Use duration (e.g., '30d', '1w', '24h') or date (e.g., '2024-12-31')."
    )


@api_token_command.command(name="create")
@click.argument(
    "name",
    nargs=1,
    type=click.STRING,
    required=True,
)
@click.option(
    "-d",
    "--description",
    type=click.STRING,
    help="Description for the API token.",
)
@click.option(
    "--scope",
    "-s",
    type=click.STRING,
    multiple=True,
    help="Scopes for the token (can be specified multiple times). "
    "Available scopes: api, read_organization, write_organization, read_cluster, write_cluster.",
)
@click.option(
    "--expires",
    type=click.STRING,
    help="Expiration time (e.g., '30d' for 30 days, '1w' for 1 week, or ISO date '2024-12-31').",
)
@click.option(
    "-o",
    "--output",
    help="Display the output of this command.",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
)
def create_api_token(
    name: str,
    description: str | None,
    scope: tuple[str, ...],
    expires: str | None,
    output: str,
) -> None:
    """Create a new API token on BentoCloud.

    \b
    Available scopes:
      - api: General API access
      - read_organization: Read organization data
      - write_organization: Write organization data
      - read_cluster: Read cluster data
      - write_cluster: Write cluster data

    \b
    Examples:
      bentoml api-token create my-token --scope api --scope read_cluster
      bentoml api-token create my-token -s api -s write_organization --expires 30d
    """
    import json as json_mod

    import yaml
    from rich.panel import Panel
    from rich.syntax import Syntax

    expired_at = _parse_expiration(expires)
    scopes = list(scope) if scope else None

    try:
        token = bentoml.api_token.create(
            name=name,
            description=description,
            scopes=scopes,
            expired_at=expired_at,
        )
    except BentoMLException as e:
        _raise_api_token_error(e, "create")

    # Display the token value prominently since it's only shown once
    if token.token:
        rich.print(
            Panel(
                f"[bold green]{token.token}[/bold green]",
                title="[bold yellow]API Token (save this - it won't be shown again!)[/bold yellow]",
                border_style="yellow",
            )
        )

    if output == "table":
        rich.print(f"\nToken [green]{token.name}[/] created successfully")
        rich.print(f"  UID: {token.uid}")
        if token.description:
            rich.print(f"  Description: {token.description}")
        if token.expired_at:
            rich.print(f"  Expires: {token.expired_at.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            rich.print("  Expires: Never")
    elif output == "json":
        info = json_mod.dumps(token.to_dict(), indent=2, default=str)
        rich.print(info)
    elif output == "yaml":
        info = yaml.dump(token.to_dict(), indent=2, sort_keys=False)
        rich.print(Syntax(info, "yaml", background_color="default"))


@api_token_command.command(name="get")
@click.argument(
    "token_uid",
    nargs=1,
    type=click.STRING,
    required=True,
)
@click.option(
    "-o",
    "--output",
    help="Display the output of this command.",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
)
def get_api_token(token_uid: str, output: str) -> None:
    """Get an API token by UID from BentoCloud."""
    import json as json_mod

    import yaml
    from rich.syntax import Syntax
    from rich.table import Table

    try:
        token = bentoml.api_token.get(token_uid=token_uid)
    except BentoMLException as e:
        _raise_api_token_error(e, "get")

    if token is None:
        raise BentoMLException(f"API token with UID '{token_uid}' not found")

    if output == "table":
        table = Table(box=None, expand=True)
        table.add_column("Field", style="bold")
        table.add_column("Value", overflow="fold")

        table.add_row("Name", token.name)
        table.add_row("UID", token.uid)
        table.add_row("Description", token.description or "-")
        table.add_row("Created At", token.created_at.strftime("%Y-%m-%d %H:%M:%S"))
        table.add_row(
            "Expired At",
            token.expired_at.strftime("%Y-%m-%d %H:%M:%S")
            if token.expired_at
            else "Never",
        )
        table.add_row(
            "Last Used At",
            token.last_used_at.strftime("%Y-%m-%d %H:%M:%S")
            if token.last_used_at
            else "Never",
        )
        table.add_row("Is Expired", str(token.is_expired))
        table.add_row("Scopes", ", ".join(token.scopes) if token.scopes else "-")
        table.add_row("Created By", token.created_by or token.user.name)
        rich.print(table)
    elif output == "json":
        info = json_mod.dumps(token.to_dict(), indent=2, default=str)
        rich.print(info)
    elif output == "yaml":
        info = yaml.dump(token.to_dict(), indent=2, sort_keys=False)
        rich.print(Syntax(info, "yaml", background_color="default"))


@api_token_command.command(name="delete")
@click.argument(
    "token_uid",
    nargs=1,
    type=click.STRING,
    required=True,
)
def delete_api_token(token_uid: str) -> None:
    """Delete an API token on BentoCloud."""
    try:
        bentoml.api_token.delete(token_uid=token_uid)
        rich.print(f"API token [green]{token_uid}[/] deleted successfully")
    except BentoMLException as e:
        _raise_api_token_error(e, "delete")


def _raise_api_token_error(err: "BentoMLException", action: str) -> "t.NoReturn":
    from http import HTTPStatus

    if err.error_code == HTTPStatus.UNAUTHORIZED:
        raise BentoMLException(
            f"{err}\n* BentoCloud API token is required for authorization. Run `bentoml cloud login` command to login"
        ) from None
    elif err.error_code == HTTPStatus.NOT_FOUND:
        raise BentoMLException("API token not found") from None
    raise BentoMLException(f"Failed to {action} API token: {err}")
