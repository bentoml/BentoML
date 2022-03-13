import logging

import click
from manager._utils import get_bin
from manager._exceptions import ManagerException
from manager._click_utils import Environment
from manager._click_utils import pass_environment

logger = logging.getLogger(__name__)


@click.command(name="login-ecr")
@click.option(
    "--stdin",
    prompt="Paste in AWS login instruction here",
    default="aws ecr get-login-password | docker login --username AWS --password-stdin aws_account_id.dkr.ecr.region.amazonaws.com",
)
@pass_environment
def main(ctx: Environment, stdin: str, *args, **kwargs) -> None:
    """
    Authenticate AWS Elastic Container Registry.
    After create a registry on AWS, paste authentication here to authenticate with docker CLI.
    """

    get_pwd_str, docker_login_str = stdin.split(" | ")
    aws, args = get_pwd_str.split(" ")[0], get_pwd_str.split(" ")[1:]
    get_pwd = get_bin(aws)[0][args]
    cmd, args = docker_login_str.split(" ")[0], docker_login_str.split(" ")[1:]
    docker_login = get_bin(cmd)[0][args]
    uri = args[-1]
    pipe = get_pwd | docker_login

    try:
        pipe()
    except Exception:
        raise ManagerException(
            "Unable to login with aws-cli. Make sure AWS CLI is setup correctly"
        )
    finally:
        import dotenv

        # write ECR registry to .env files
        # Write changes to .env file.
        dotenv.set_key(dotenv.find_dotenv(), "AWS_URL", uri)
        logger.info(
            ":smile: [bold green]Successfully logged into ECR and updated AWS_URL[/]",
            extra={"markup": True},
        )
