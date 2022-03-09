# type: ignore[reportUnusedFunction]
import logging

import click

from bentoml.exceptions import CLIException

from ..cli.click_utils import BentoMLCommandGroup
from ..yatai_rest_api_client.config import add_context
from ..yatai_rest_api_client.config import update_context
from ..yatai_rest_api_client.config import YataiClientContext
from ..yatai_rest_api_client.config import default_context_name

logger = logging.getLogger(__name__)


def add_login_command(cli: click.Group) -> None:
    @cli.group(name="yatai", cls=BentoMLCommandGroup)
    def yatai_cli():
        """Yatai Subcommands Groups"""

    @yatai_cli.command(help="Login to Yatai server")
    @click.option(
        "--endpoint", type=click.STRING, help="Yatai endpoint, i.e: https://yatai.com"
    )
    @click.option("--api-token", type=click.STRING, help="Yatai user API token")
    def login(endpoint: str, api_token: str) -> None:
        if not endpoint:
            raise CLIException("need --endpoint")

        if not api_token:
            raise CLIException("need --api-token")

        ctx = YataiClientContext(
            name=default_context_name,
            endpoint=endpoint,
            api_token=api_token,
        )

        add_context(ctx)

        yatai_rest_client = ctx.get_yatai_rest_api_client()

        user = yatai_rest_client.get_current_user()
        org = yatai_rest_client.get_current_organization()

        if org is None:
            raise CLIException("current organization is not found")

        logger.info(
            f"login successfully! user: {user.name}, organization: {org.name}",
        )
