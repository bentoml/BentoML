from .bento_management import add_bento_sub_command
from .bento_service import create_bento_service_cli
from .deployment import get_deployment_sub_command

# from .yatai_service import add_yatai_service_sub_command


def create_bentoml_cli():
    # pylint: disable=unused-variable

    _cli = create_bento_service_cli()

    # Commands created here aren't mean to be used from generated BentoService CLI when
    # installed as PyPI package. The are only used as part of BentoML cli command.

    deployment_sub_command = get_deployment_sub_command()
    add_bento_sub_command(_cli)
    # add_yatai_service_sub_command(_cli)
    _cli.add_command(deployment_sub_command)

    return _cli


cli = create_bentoml_cli()

if __name__ == "__main__":
    cli()
