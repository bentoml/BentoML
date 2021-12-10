import click

from bentoml.exceptions import CLIException
from bentoml._internal.cli.click_utils import _echo
from bentoml._internal.cli.click_utils import CLI_COLOR_SUCCESS
from bentoml._internal.yatai_rest_api_client.config import add_context
from bentoml._internal.yatai_rest_api_client.config import YataiClientContext
from bentoml._internal.yatai_rest_api_client.config import default_context_name


def add_login_command(cli):
    @cli.group(name="yatai")
    def yatai_cli():
        """yatai sub commands"""

    @yatai_cli.command(help="Login to yatai server")
    @click.option(
        "--endpoint", type=click.STRING, help="Yatai endpoint, like https://yatai.ai"
    )
    @click.option("--api-token", type=click.STRING, help="Yatai user api token")
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
        _echo(
            f"login successfully! user: {user.name}, organization: {org.name}",
            color=CLI_COLOR_SUCCESS,
        )
