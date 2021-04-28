# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from bentoml.cli.aws_lambda import get_aws_lambda_sub_command
from bentoml.cli.aws_sagemaker import get_aws_sagemaker_sub_command
from bentoml.cli.azure_functions import get_azure_functions_sub_command
from bentoml.cli.aws_ec2 import get_aws_ec2_sub_command
from bentoml.cli.bento_management import add_bento_sub_command
from bentoml.cli.bento_service import create_bento_service_cli
from bentoml.cli.yatai_service import add_yatai_service_sub_command
from bentoml.cli.deployment import get_deployment_sub_command


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


if __name__ == "__main__":
    from bentoml import commandline_interface

    commandline_interface()  # pylint: disable=no-value-for-parameter
