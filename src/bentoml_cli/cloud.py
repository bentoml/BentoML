from __future__ import annotations

import click
import json
import click_option_group as cog


def add_cloud_command(cli: click.Group) -> click.Group:
    from bentoml_cli.utils import BentoMLCommandGroup
    from bentoml.exceptions import CLIException
    from bentoml._internal.configuration import get_quiet_mode
    from bentoml._internal.utils import bentoml_cattr
    from bentoml._internal.cloud.client import RestApiClient
    from bentoml._internal.cloud.config import add_context
    from bentoml._internal.cloud.config import CloudClientContext
    from bentoml._internal.cloud.config import default_context_name
    from bentoml._internal.cloud.config import CloudClientConfig

    @cli.group(name="cloud", aliases=["yatai"], cls=BentoMLCommandGroup)
    def cloud():
        """BentoCloud Subcommands Groups

        \b
        Note that 'bentoml yatai' is mainly for backward compatible reason. It is recommended
        to use 'bentoml cloud'.
        """

    @cloud.command()
    @cog.optgroup.group(
        "Login", help="Required login options", cls=cog.RequiredAllOptionGroup
    )
    @cog.optgroup.option(
        "--endpoint",
        type=click.STRING,
        help="BentoCloud or Yatai endpoint, i.e: https://cloud.bentoml.com",
    )
    @cog.optgroup.option(
        "--api-token",
        type=click.STRING,
        help="BentoCloud or Yatai user API token",
    )
    @click.option(
        "--context",
        type=click.STRING,
        help="BentoCloud or Yatai context name for the endpoint and API token",
        default=default_context_name,
    )
    def login(endpoint: str, api_token: str, context: str) -> None:  # type: ignore (not accessed)
        """Login to BentoCloud or Yatai server."""
        cloud_rest_client = RestApiClient(endpoint, api_token)
        user = cloud_rest_client.get_current_user()

        if user is None:
            raise CLIException("current user is not found")

        org = cloud_rest_client.get_current_organization()

        if org is None:
            raise CLIException("current organization is not found")

        ctx = CloudClientContext(
            name=context,
            endpoint=endpoint,
            api_token=api_token,
            email=user.email,
        )

        add_context(ctx)
        click.echo(
            f"Successfully logged in to BentoCloud for {user.name} in {org.name}"
        )

    @cloud.command(aliases=["current-context"])
    def get_current_context() -> None:  # type: ignore (not accessed)
        """Get current cloud context."""
        cur = CloudClientConfig.get_config().get_current_context()
        if not get_quiet_mode():
            click.echo(json.dumps(bentoml_cattr.unstructure(cur), indent=2))

    @cloud.command()
    @click.argument("context", type=click.STRING)
    def update_current_context(context: str) -> None:  # type: ignore (not accessed)
        """Update current context"""
        ctx = CloudClientConfig.get_config().set_current_context(context)
        click.echo(f"Successfully switched to context: {ctx.name}")

    return cli
