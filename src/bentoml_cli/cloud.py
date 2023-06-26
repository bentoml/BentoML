from __future__ import annotations

import click


def add_login_command(cli: click.Group) -> None:
    from bentoml_cli.utils import BentoMLCommandGroup
    from bentoml.exceptions import CLIException
    from bentoml._internal.cloud.client import CloudRESTApiClient
    from bentoml._internal.cloud.config import add_context
    from bentoml._internal.cloud.config import CloudClientContext
    from bentoml._internal.cloud.config import default_context_name

    @cli.group(name="cloud", aliases=["yatai"], cls=BentoMLCommandGroup)
    def cloud_cli():
        """BentoCloud Subcommands Groups

        \b
        Note that 'bentoml yatai' is mainly for backward compatible reason. It is recommended
        to use 'bentoml cloud'.
        """

    @cloud_cli.command()
    @click.option(
        "--endpoint",
        type=click.STRING,
        help="BentoCloud or Yatai endpoint, i.e: https://cloud.bentoml.com",
    )
    @click.option(
        "--api-token", type=click.STRING, help="BentoCloud or Yatai user API token"
    )
    @click.option(
        "--context",
        type=click.STRING,
        help="BentoCloud or Yatai context name for the endpoint and API token",
        default=default_context_name,
    )
    def login(endpoint: str, api_token: str, context: str) -> None:  # type: ignore (not accessed)
        """Login to BentoCloud or Yatai server."""
        if not endpoint:
            raise CLIException("need --endpoint")

        if not api_token:
            raise CLIException("need --api-token")

        cloud_rest_client = CloudRESTApiClient(endpoint, api_token)
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
            f'Successfully logged in as user "{user.name}" in organization "{org.name}".'
        )
