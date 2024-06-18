from __future__ import annotations

import http.server
import json
import socket
import threading
import typing as t
import webbrowser

import click
import click_option_group as cog
from InquirerPy import inquirer

from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.config import CloudClientConfig
from bentoml._internal.cloud.config import CloudClientContext
from bentoml._internal.cloud.config import add_context
from bentoml._internal.cloud.config import default_context_name
from bentoml._internal.utils import bentoml_cattr
from bentoml.exceptions import CLIException
from bentoml.exceptions import CloudRESTApiClientError
from bentoml_cli.utils import BentoMLCommandGroup

if t.TYPE_CHECKING:
    from .utils import SharedOptions
    
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        return s.getsockname()[1]


# 定义回调处理程序
class CallbackHandler(http.server.SimpleHTTPRequestHandler):
    verification_code = None
    server_instance = None

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)

        if data.get("verification_code") == self.verification_code:
            token = data.get("api_token")
            print(f"Received token: {token}")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Login successful! You can close this window.")
            self.server.token = token  # 将令牌存储在服务器实例中
            # 关闭服务器
            def shutdown_server(server):
                server.shutdown()
                server.server_close()
            threading.Thread(target=shutdown_server, args=(self.server_instance,)).start()
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid verification code")

@click.group(name="cloud", cls=BentoMLCommandGroup)
def cloud_command():
    """BentoCloud Subcommands Groups."""


@cloud_command.command()
@cog.optgroup.group(
    "Login", help="Required login options", cls=cog.RequiredAllOptionGroup
)
@cog.optgroup.option(
    "--endpoint",
    type=click.STRING,
    help="BentoCloud or Yatai endpoint, default as https://cloud.bentoml.com",
    default="https://cloud.bentoml.com",
)
@cog.optgroup.option(
    "--api-token",
    type=click.STRING,
    help="BentoCloud or Yatai user API token",
)
@click.pass_obj
def login(shared_options: SharedOptions, endpoint: str, api_token: str) -> None:  # type: ignore (not accessed)
    """Login to BentoCloud or Yatai server."""
    if not api_token:
        choice = inquirer.select(
            message="How would you like to authenticate BentoML CLI? [Use arrows to move]",
            choices=[
                {"name": "Create a new API token with a web browser", "value": "create"},
                {"name": "Paste an existing API token", "value": "paste"},
            ],
        ).execute()
        
        if choice == "create":
            port = find_free_port()
            createURL = f'{endpoint}/api-tokens/new?callback=http://localhost:{port}'
            click.echo(f'ℹ You can generate an API Token at {createURL}')
            ## Press Enter to open https://cloud.bentoml.com/api_tokens/new in your browser..
            # 提示用户按下回车键以继续
            input("Press Enter to open {} in your browser...".format(createURL))
            ## define a handler 
            ## start a server
            ## open webbrowser
            if webbrowser.open_new_tab(createURL):
                click.echo(f"✓ Opened {createURL} in your web browser.")
            else:
                click.echo(f"✗ Failed to open browser. Try create a new API token at {endpoint}/api_tokens/")
            return
        elif choice == "paste":
            api_token = click.prompt("? Paste your authentication token", type=str, hide_input=True)
    try:
        cloud_rest_client = RestApiClient(endpoint, api_token)
        user = cloud_rest_client.v1.get_current_user()

        if user is None:
            raise CLIException("current user is not found")

        org = cloud_rest_client.v1.get_current_organization()

        if org is None:
            raise CLIException("current organization is not found")

        ctx = CloudClientContext(
            name=shared_options.cloud_context
            if shared_options.cloud_context is not None
            else default_context_name,
            endpoint=endpoint,
            api_token=api_token,
            email=user.email,
        )

        add_context(ctx)
        click.echo(f"✓ Configured BentoCloud credentials (current-context: {ctx.name})")
        click.echo(f"✓ Logged in as {user.email} at {org.name}")
    except CloudRESTApiClientError  as e:
        if e.error_code == 401:
            click.echo(f"✗ Error validating token: HTTP 401: Bad credentials ({endpoint}/api-token)")
        else:
            click.echo(f"✗ Error validating token: HTTP {e.error_code}")

@cloud_command.command()
def current_context() -> None:  # type: ignore (not accessed)
    """Get current cloud context."""
    click.echo(
        json.dumps(
            bentoml_cattr.unstructure(
                CloudClientConfig.get_config().get_current_context()
            ),
            indent=2,
        )
    )


@cloud_command.command()
def list_context() -> None:  # type: ignore (not accessed)
    """List all available context."""
    config = CloudClientConfig.get_config()
    click.echo(
        json.dumps(
            bentoml_cattr.unstructure([i.name for i in config.contexts]), indent=2
        )
    )


@cloud_command.command()
@click.argument("context_name", type=click.STRING)
def update_current_context(context_name: str) -> None:  # type: ignore (not accessed)
    """Update current context"""
    ctx = CloudClientConfig.get_config().set_current_context(context_name)
    click.echo(f"Successfully switched to context: {ctx.name}")
