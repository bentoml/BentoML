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

import click

from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.utils.lazy_loader import LazyLoader
from bentoml.cli.utils import Spinner
from bentoml.utils import get_default_yatai_client
from bentoml.cli.click_utils import (
    BentoMLCommandGroup,
    parse_bento_tag_callback,
    _echo,
    CLI_COLOR_SUCCESS,
    parse_labels_callback,
)
from bentoml.cli.deployment import (
    _print_deployment_info,
    _print_deployments_info,
)
from bentoml.yatai.deployment import ALL_NAMESPACE_TAG
from bentoml.exceptions import CLIException

yatai_proto = LazyLoader("yatai_proto", globals(), "bentoml.yatai.proto")


def get_aws_ec2_sub_command():
    @click.group(name="ec2", cls=BentoMLCommandGroup, help="commandas for ec2")
    def aws_ec2():
        pass

    @aws_ec2.command(help="Deploy BentoServide to ec2")
    @click.argument("name", type=click.STRING)
    @click.option("-b", "--bento", type=click.STRING, callback=parse_bento_tag_callback)
    def deploy(name, bento):
        yatai_client = get_default_yatai_client()
        bento_name, bento_version = bento.split(":")
        result = yatai_client.deployment.create_ec2_deployment(
            name=name, bento_name=bento_name, bento_version=bento_version
        )

        _echo(f"Successfully created AWS EC2 deployment ", CLI_COLOR_SUCCESS)

    @aws_ec2.command(help="Delete AWS EC2 deployment")
    @click.argument("name", type=click.STRING)
    @click.option(
        "-n",
        "--namespace",
        type=click.STRING,
        help='Deployment namespace managed by BentoML, default value is "dev" which'
        "can be changed in BentoML configuration yatai_service/default_namespace",
    )
    @click.option(
        "--force",
        is_flag=True,
        help="force delete the deployment record in database and "
        "ignore errors when deleting cloud resources",
    )
    def delete(name, namespace, force):
        yatai_client = get_default_yatai_client()
        get_deployment_result = yatai_client.deployment.get(
            namespace=namespace, name=name
        )
        if get_deployment_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_deployment_result.status
            )
            raise CLIException(f"{error_code}:{error_message}")

        delete_deployment_result = yatai_client.deployment.delete(
            namespace=namespace, deployment_name=name, force_delete=force
        )
        if (
            delete_deployment_result.status.status_code
            != yatai_proto.status_pb2.Status.OK
        ):
            error_code, error_message = status_pb_to_error_code_and_message(
                get_deployment_result.status
            )
            raise CLIException(f"{error_code}:{error_message}")

        _echo(f"Successfiully deleted AWS EC2 deployment '{name}'", CLI_COLOR_SUCCESS)

    return aws_ec2
