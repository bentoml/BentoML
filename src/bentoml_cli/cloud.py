from __future__ import annotations

import logging
import sys
import time
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
