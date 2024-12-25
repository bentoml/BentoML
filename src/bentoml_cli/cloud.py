from __future__ import annotations

import contextlib
import sys
import urllib.parse
import webbrowser

import click
import rich

from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.config import DEFAULT_ENDPOINT
from bentoml._internal.cloud.config import CloudClientConfig
from bentoml._internal.cloud.config import CloudClientContext
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.utils import reserve_free_port
from bentoml._internal.utils.cattr import bentoml_cattr
from bentoml.exceptions import CLIException
from bentoml.exceptions import CloudRESTApiClientError
from bentoml_cli.auth_server import AuthCallbackHttpServer
from bentoml_cli.utils import BentoMLCommandGroup


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
    import questionary
    from rich.prompt import Confirm

    if not api_token:
        choice = questionary.select(
            message="How would you like to authenticate BentoML CLI? [Use arrows to move]",
            choices=[
                {
                    "name": "Create a new API token with a web browser",
                    "value": "create",
                },
                {"name": "Paste an existing API token", "value": "paste"},
            ],
        ).ask()

        if choice == "create":
            with contextlib.ExitStack() as port_stack:
                port = port_stack.enter_context(
                    reserve_free_port(enable_so_reuseport=True)
                )
            callback_server = AuthCallbackHttpServer(port)
            endpoint = endpoint.rstrip("/")
            baseURL = f"{endpoint}/api_tokens"
            encodedCallback = urllib.parse.quote(callback_server.callback_url)
            authURL = f"{baseURL}?callback={encodedCallback}"
            if Confirm.ask(
                f"Would you like to open [blue]{authURL}[/] in your browser?",
                default=True,
            ):
                if webbrowser.open_new_tab(authURL):
                    rich.print(
                        f":white_check_mark: Opened [blue]{authURL}[/] in your web browser."
                    )
                else:
                    rich.print(
                        f":police_car_light: Failed to open browser. Try create a new API token at {baseURL} or Open [blue]{authURL}[/] yourself"
                    )
            else:
                rich.print(
                    f":backhand_index_pointing_right: Open [blue]{authURL}[/] yourself..."
                )
            try:
                rich.print(":hourglass: Waiting for authentication...")
                code = callback_server.wait_indefinitely_for_code()
                if code is None:
                    raise ValueError(
                        "No code could be obtained from browser callback page"
                    )
                api_token = code
            except Exception:
                rich.print(":police_car_light: Error accquiring token from web browser")
                return
        elif choice == "paste":
            api_token = click.prompt(
                "? Paste your authentication token", type=str, hide_input=True
            )
    try:
        cloud_rest_client = RestApiClient(endpoint, api_token)
        user = cloud_rest_client.v1.get_current_user()

        if user is None:
            raise CLIException("current user is not found")

        org = cloud_rest_client.v1.get_current_organization()

        if org is None:
            raise CLIException("current organization is not found")

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
