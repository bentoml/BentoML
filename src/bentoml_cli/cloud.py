from __future__ import annotations

import logging
import sys
import time
import typing as t
import urllib.parse
import webbrowser

import click
import httpx
import rich

from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.config import DEFAULT_ENDPOINT
from bentoml._internal.cloud.config import CloudClientConfig
from bentoml._internal.cloud.config import CloudClientContext
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.utils.cattr import bentoml_cattr
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import CLIException
from bentoml.exceptions import CloudRESTApiClientError
from bentoml_cli.utils import BentoMLCommandGroup

logger = logging.getLogger("bentoml.cli")


@click.group(name="cloud", cls=BentoMLCommandGroup)
def cloud_command():
    """BentoCloud Subcommands Groups."""


@cloud_command.command()
@click.option(
    "--endpoint",
    type=click.STRING,
    help="BentoCloud endpoint",
    default=DEFAULT_ENDPOINT,
    envvar="BENTO_CLOUD_API_ENDPOINT",
    show_default=True,
    show_envvar=True,
    required=False,
)
@click.option(
    "--api-token",
    type=click.STRING,
    help="BentoCloud user API token",
    envvar="BENTO_CLOUD_API_KEY",
    show_envvar=True,
    required=False,
)
def login(endpoint: str, api_token: str) -> None:  # type: ignore (not accessed)
    """Login to BentoCloud."""
    from rich.prompt import Confirm
    from rich.prompt import Prompt
    from rich_toolkit.menu import Menu

    if not api_token:
        choice = Menu(
            label="How would you like to authenticate BentoML CLI?",
            options=[
                {
                    "name": "Create a new API token with a web browser",
                    "value": "create",
                },
                {"name": "Paste an existing API token", "value": "paste"},
            ],
        ).ask()

        if choice == "create":
            endpoint = endpoint.rstrip("/")
            code_url = f"{endpoint}/api/v1/auth/code"
            token_url = f"{endpoint}/api/v1/auth/token"
            try:
                code = httpx.get(code_url).json()["code"]
            except httpx.HTTPError as e:
                rich.print(
                    f":police_car_light: Error fetching auth code: {e}", file=sys.stderr
                )
                raise SystemExit(1)
            query = urllib.parse.urlencode({"code": code})
            auth_url = f"{endpoint}/api_tokens?{query}"
            if Confirm.ask(
                f"Would you like to open [blue]{auth_url}[/] in your browser?",
                default=True,
            ):
                if webbrowser.open_new_tab(auth_url):
                    rich.print(
                        f":white_check_mark: Opened [blue]{auth_url}[/] in your web browser."
                    )
                else:
                    rich.print(
                        f":police_car_light: Failed to open browser. Try creating a new API token at [blue]{auth_url}[/] yourself"
                    )
            else:
                rich.print(
                    f":backhand_index_pointing_right: Open [blue]{auth_url}[/] yourself..."
                )
            rich.print(":hourglass: Waiting for authentication...")
            while True:
                resp = httpx.get(token_url, params={"code": code})
                if resp.is_success:
                    api_token = resp.json()["token"]
                    break
                logger.debug(
                    "Failed to obtain token(%s): %s", resp.status_code, resp.text
                )
                time.sleep(1)
        elif choice == "paste":
            api_token = Prompt.ask(":key: Paste your API token", password=True)
    try:
        cloud_rest_client = RestApiClient(endpoint, api_token)
        user = cloud_rest_client.v1.get_current_user()

        if user is None:
            raise CLIException("The current user is not found")

        org = cloud_rest_client.v1.get_current_organization()

        if org is None:
            raise CLIException("The current organization is not found")

        current_context_name = CloudClientConfig.get_config().current_context_name
        cloud_context = BentoMLContainer.cloud_context.get()

        ctx = CloudClientContext(
            name=cloud_context if cloud_context is not None else current_context_name,
            endpoint=endpoint,
            api_token=api_token,
            email=user.email,
        )

        ctx.save()
        rich.print(
            f":white_check_mark: Configured BentoCloud credentials (current-context: {ctx.name})"
        )
        rich.print(
            f":white_check_mark: Logged in as [blue]{user.email}[/] at [blue]{org.name}[/] organization"
        )
    except CloudRESTApiClientError as e:
        if e.error_code == 401:
            rich.print(
                f":police_car_light: Error validating token: HTTP 401: Bad credentials ({endpoint}/api-token)",
                file=sys.stderr,
            )
        else:
            rich.print(
                f":police_car_light: Error validating token: HTTP {e.error_code}",
                file=sys.stderr,
            )
        raise SystemExit(1)


@cloud_command.command()
def current_context() -> None:  # type: ignore (not accessed)
    """Get current cloud context."""
    rich.print_json(
        data=bentoml_cattr.unstructure(CloudClientConfig.get_config().get_context())
    )


@cloud_command.command()
def list_context() -> None:  # type: ignore (not accessed)
    """List all available context."""
    config = CloudClientConfig.get_config()
    rich.print_json(data=bentoml_cattr.unstructure([i.name for i in config.contexts]))


@cloud_command.command()
@click.argument("context_name", type=click.STRING)
def update_current_context(context_name: str) -> None:  # type: ignore (not accessed)
    """Update current context"""
    ctx = CloudClientConfig.get_config().set_current_context(context_name)
    rich.print(f"Successfully switched to context: {ctx.name}")


# API Token management subcommands
@cloud_command.group(name="api-token", cls=BentoMLCommandGroup)
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
) -> None:  # type: ignore (not accessed)
    """List all API tokens on BentoCloud."""
    import json as json_mod

    import yaml
    from rich.syntax import Syntax
    from rich.table import Table
    from simple_di import Provide
    from simple_di import inject

    from bentoml._internal.cloud import BentoCloudClient

    @inject
    def _list(
        _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    ) -> None:
        try:
            tokens = _cloud_client.api_token.list(search=search)
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

    _list()


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
) -> None:  # type: ignore (not accessed)
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
      bentoml cloud api-token create my-token --scope api --scope read_cluster
      bentoml cloud api-token create my-token -s api -s write_organization --expires 30d
    """
    import json as json_mod
    from datetime import datetime
    from datetime import timedelta

    import yaml
    from rich.panel import Panel
    from rich.syntax import Syntax
    from simple_di import Provide
    from simple_di import inject

    from bentoml._internal.cloud import BentoCloudClient

    def parse_expiration(expires_str: str | None) -> datetime | None:
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

    @inject
    def _create(
        _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    ) -> None:
        expired_at = parse_expiration(expires)
        scopes = list(scope) if scope else None

        try:
            token = _cloud_client.api_token.create(
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
                rich.print(
                    f"  Expires: {token.expired_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            else:
                rich.print("  Expires: Never")
        elif output == "json":
            info = json_mod.dumps(token.to_dict(), indent=2, default=str)
            rich.print(info)
        elif output == "yaml":
            info = yaml.dump(token.to_dict(), indent=2, sort_keys=False)
            rich.print(Syntax(info, "yaml", background_color="default"))

    _create()


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
def get_api_token(token_uid: str, output: str) -> None:  # type: ignore (not accessed)
    """Get an API token by UID from BentoCloud."""
    import json as json_mod

    import yaml
    from rich.syntax import Syntax
    from rich.table import Table
    from simple_di import Provide
    from simple_di import inject

    from bentoml._internal.cloud import BentoCloudClient

    @inject
    def _get(
        _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    ) -> None:
        try:
            token = _cloud_client.api_token.get(token_uid=token_uid)
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

    _get()


@api_token_command.command(name="delete")
@click.argument(
    "token_uid",
    nargs=1,
    type=click.STRING,
    required=True,
)
def delete_api_token(token_uid: str) -> None:  # type: ignore (not accessed)
    """Delete an API token on BentoCloud."""
    from simple_di import Provide
    from simple_di import inject

    from bentoml._internal.cloud import BentoCloudClient

    @inject
    def _delete(
        _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    ) -> None:
        try:
            _cloud_client.api_token.delete(token_uid=token_uid)
            rich.print(f"API token [green]{token_uid}[/] deleted successfully")
        except BentoMLException as e:
            _raise_api_token_error(e, "delete")

    _delete()


def _raise_api_token_error(err: "BentoMLException", action: str) -> "t.NoReturn":
    from http import HTTPStatus

    if err.error_code == HTTPStatus.UNAUTHORIZED:
        raise BentoMLException(
            f"{err}\n* BentoCloud API token is required for authorization. Run `bentoml cloud login` command to login"
        ) from None
    elif err.error_code == HTTPStatus.NOT_FOUND:
        raise BentoMLException("API token not found") from None
    raise BentoMLException(f"Failed to {action} API token: {err}")
