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

import re
import os
import json
import click
import tempfile
import subprocess
import multiprocessing
from pathlib import Path

import psutil
from ruamel.yaml import YAML

from bentoml.saved_bundle import (
    load,
    load_bento_service_api,
    load_saved_bundle_config,
    load_bento_service_metadata,
)
from bentoml.cli.aws_lambda import get_aws_lambda_sub_command
from bentoml.cli.aws_sagemaker import get_aws_sagemaker_sub_command
from bentoml.cli.azure_functions import get_azure_functions_sub_command
from bentoml.cli.bento_management import add_bento_sub_command
from bentoml.cli.bento_service import create_bento_service_cli
from bentoml.cli.yatai_service import add_yatai_service_sub_command
from bentoml.server import BentoAPIServer
from bentoml.server.open_api import get_open_api_spec_json
from bentoml.server.utils import get_gunicorn_num_of_workers
from bentoml.marshal import MarshalService
from bentoml.cli.click_utils import BentoMLCommandGroup, conditional_argument, _echo
from bentoml.cli.deployment import get_deployment_sub_command
from bentoml.cli.config import get_configuration_sub_command
from bentoml.utils import ProtoMessageToDict, reserve_free_port
from bentoml.utils.s3 import is_s3_url
from bentoml.yatai.client import YataiClient
from bentoml.yatai.proto import status_pb2
from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.exceptions import BentoMLException

try:
    import click_completion

    click_completion.init()
    shell_types = click_completion.DocumentedChoice(click_completion.core.shells)
except ImportError:
    # click_completion package is optional to use BentoML cli,
    click_completion = None
    shell_types = click.Choice(['bash', 'zsh', 'fish', 'powershell'])


def escape_shell_params(param):
    k, v = param.split('=')
    v = re.sub(r'([^a-zA-Z0-9])', r'\\\1', v)
    return '{}={}'.format(k, v)


def run_with_conda_env(bundle_path, command):
    config = load_saved_bundle_config(bundle_path)
    metadata = config['metadata']
    env_name = metadata['service_name'] + '_' + metadata['service_version']

    yaml = YAML()
    yaml.default_flow_style = False
    tmpf = tempfile.NamedTemporaryFile(delete=False)
    env_path = tmpf.name + '.yaml'
    yaml.dump(config['env']['conda_env'], Path(env_path))

    pip_req = os.path.join(bundle_path, 'requirements.txt')

    subprocess.call(
        'command -v conda >/dev/null 2>&1 || {{ echo >&2 "--with-conda '
        'parameter requires conda but it\'s not installed."; exit 1; }} && '
        'conda env update -n {env_name} -f {env_file} && '
        'conda init bash && '
        'eval "$(conda shell.bash hook)" && '
        'conda activate {env_name} && '
        '{{ [ -f {pip_req} ] && pip install -r {pip_req} || echo "no pip '
        'dependencies."; }} && {cmd}'.format(
            env_name=env_name, env_file=env_path, pip_req=pip_req, cmd=command,
        ),
        shell=True,
    )
    return


def create_bentoml_cli():
    # pylint: disable=unused-variable

    _cli = create_bento_service_cli()

    # Commands created here aren't mean to be used from generated BentoService CLI when
    # installed as PyPI package. The are only used as part of BentoML cli command.

    config_sub_command = get_configuration_sub_command()
    aws_sagemaker_sub_command = get_aws_sagemaker_sub_command()
    aws_lambda_sub_command = get_aws_lambda_sub_command()
    deployment_sub_command = get_deployment_sub_command()
    azure_function_sub_command = get_azure_functions_sub_command()
    add_bento_sub_command(_cli)
    add_yatai_service_sub_command(_cli)
    _cli.add_command(config_sub_command)
    _cli.add_command(aws_sagemaker_sub_command)
    _cli.add_command(aws_lambda_sub_command)
    _cli.add_command(azure_function_sub_command)
    _cli.add_command(deployment_sub_command)

    return _cli


cli = create_bentoml_cli()

if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
