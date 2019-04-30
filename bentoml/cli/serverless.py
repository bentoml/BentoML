# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import shutil
import subprocess

from packaging import version

from bentoml.utils import Path
from bentoml.cli.whichcraft import which
from bentoml.cli.aws_lambda_template import AWS_HANDLER_PY_TEMPLATE, \
     update_serverless_configuration_for_aws
from bentoml.cli.gcp_function_template import GOOGLE_MAIN_PY_TEMPLATE, \
     update_serverless_configuration_for_google
from bentoml.cli.utils import generate_bentoml_deployment_snapshot_path

SERVERLESS_PROVIDER = {
    'aws-lambda': 'aws-python3',
    'aws-lambda-py2': 'aws-python',
    'gcp-function': 'google-python',
}


def generate_base_serverless_files(output_path, platform, name):
    subprocess.call(
        ['serverless', 'create', '--template', platform, '--name', name], cwd=output_path)
    if platform != 'google-python':
        subprocess.call(['serverless', 'plugin', 'install', '-n', 'serverless-python-requirements'],
                        cwd=output_path)
        subprocess.call(['serverless', 'plugin', 'install', '-n', 'serverless-apigw-binary'],
                        cwd=output_path)
    return


def deploy_serverless_file(output_path):
    subprocess.call(['serverless', 'deploy'], cwd=output_path)
    return


def add_model_service_archive(bento_service, archive_path, output_path):
    model_serivce_archive_path = os.path.join(output_path, bento_service.name)
    shutil.copytree(archive_path, model_serivce_archive_path)
    return


def generate_handler_py(bento_service, output_path, platform):
    api = bento_service.get_service_apis()[0]
    if platform == 'google-python':
        file_name = 'main.py'
        handler_py_content = GOOGLE_MAIN_PY_TEMPLATE.format(class_name=bento_service.name,
                                                            api_name=api.name)
    else:
        file_name = 'handler.py'
        handler_py_content = AWS_HANDLER_PY_TEMPLATE.format(class_name=bento_service.name,
                                                            api_name=api.name)

    handler_file = os.path.join(output_path, file_name)

    with open(handler_file, 'w') as f:
        f.write(handler_py_content)
    return


def check_serverless_compatiable_version():
    if which('serverless') is None:
        raise ValueError(
            'Serverless framework is not installed, please visit ' +
            'www.serverless.com for install instructions.'
        )

    version_result = subprocess.check_output(['serverless', '-v'])
    parsed_version = version.parse(version_result.decode('utf-8').strip())

    if parsed_version >= version.parse('1.40.0'):
        return
    else:
        raise ValueError(
            'Incompatiable serverless version, please install version 1.40.0 or greater')


def generate_serverless_bundle(bento_service, platform, archive_path, additional_options):
    check_serverless_compatiable_version()

    provider = SERVERLESS_PROVIDER[platform]
    output_path = generate_bentoml_deployment_snapshot_path(bento_service.name, platform)
    Path(output_path).mkdir(parents=True, exist_ok=False)


    serverless_config_file = os.path.join(output_path, 'serverless.yml')
    generate_base_serverless_files(output_path, provider, bento_service.name)

    if provider != 'google-python':
        update_serverless_configuration_for_aws(bento_service, serverless_config_file,
                                                additional_options)
    else:
        update_serverless_configuration_for_google(bento_service, serverless_config_file,
                                                   additional_options)

    generate_handler_py(bento_service, output_path, provider)

    shutil.copy(os.path.join(archive_path, 'requirements.txt'), output_path)
    add_model_service_archive(bento_service, archive_path, output_path)

    return os.path.realpath(output_path)


def deploy_with_serverless(bento_service, platform, archive_path, extra_args):
    output_path = generate_serverless_bundle(bento_service, platform, archive_path, extra_args)
    deploy_serverless_file(output_path)
    return output_path
