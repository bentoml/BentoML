from bentoml.cli.aws_ec2 import get_aws_ec2_sub_command
from bentoml.cli.aws_lambda import get_aws_lambda_sub_command
from bentoml.cli.aws_sagemaker import get_aws_sagemaker_sub_command
from bentoml.cli.azure_functions import get_azure_functions_sub_command
from bentoml.cli.bento_management import add_bento_sub_command
from bentoml.cli.bento_service import create_bento_service_cli
from bentoml.cli.deployment import get_deployment_sub_command
from bentoml.cli.yatai_service import add_yatai_service_sub_command


def create_bentoml_cli():
    # pylint: disable=unused-variable

    _cli = create_bento_service_cli()

    # Commands created here aren't mean to be used from generated BentoService CLI when
    # installed as PyPI package. The are only used as part of BentoML cli command.

    aws_sagemaker_sub_command = get_aws_sagemaker_sub_command()
    aws_lambda_sub_command = get_aws_lambda_sub_command()
    deployment_sub_command = get_deployment_sub_command()
    azure_function_sub_command = get_azure_functions_sub_command()
    aws_ec2_sub_command = get_aws_ec2_sub_command()
    add_bento_sub_command(_cli)
    add_yatai_service_sub_command(_cli)
    _cli.add_command(aws_sagemaker_sub_command)
    _cli.add_command(aws_lambda_sub_command)
    _cli.add_command(aws_ec2_sub_command)
    _cli.add_command(azure_function_sub_command)
    _cli.add_command(deployment_sub_command)

    return _cli


cli = create_bentoml_cli()

if __name__ == "__main__":
    cli()
