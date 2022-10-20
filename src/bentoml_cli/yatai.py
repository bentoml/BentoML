from __future__ import annotations

import logging

import click

logger = logging.getLogger("bentoml")


def add_login_command(cli: click.Group) -> None:
    from bentoml_cli.utils import BentoMLCommandGroup
    from bentoml.exceptions import CLIException
    from bentoml._internal.yatai_rest_api_client.yatai import YataiRESTApiClient
    from bentoml._internal.yatai_rest_api_client.config import add_context
    from bentoml._internal.yatai_rest_api_client.config import YataiClientContext
    from bentoml._internal.yatai_rest_api_client.config import default_context_name

    @cli.group(name="yatai", cls=BentoMLCommandGroup)
    def yatai_cli():
        """Yatai Subcommands Groups"""

    @yatai_cli.command()
    @click.option(
        "--endpoint", type=click.STRING, help="Yatai endpoint, i.e: https://yatai.com"
    )
    @click.option("--api-token", type=click.STRING, help="Yatai user API token")
    def login(endpoint: str, api_token: str) -> None:  # type: ignore (not accessed)
        """Login to Yatai server."""
        if not endpoint:
            raise CLIException("need --endpoint")

        if not api_token:
            raise CLIException("need --api-token")

        yatai_rest_client = YataiRESTApiClient(endpoint, api_token)
        user = yatai_rest_client.get_current_user()

        if user is None:
            raise CLIException("current user is not found")

        org = yatai_rest_client.get_current_organization()

        if org is None:
            raise CLIException("current organization is not found")

        ctx = YataiClientContext(
            name=default_context_name,
            endpoint=endpoint,
            api_token=api_token,
            email=user.email,
        )

        add_context(ctx)

        logger.info(
            'Successfully logged in as user "%s" in organization "%s".',
            user.name,
            org.name,
        )
