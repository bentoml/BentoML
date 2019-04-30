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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import subprocess

from packaging import version

from bentoml.archive import load
from bentoml.utils import Path
from bentoml.utils.whichcraft import which
from bentoml.utils.exceptions import BentoMLException
from bentoml.deployment.aws_lambda_template import update_aws_lambda_configuration
from bentoml.deployment.gcp_function_template import update_gcp_function_configuration
from bentoml.deployment.utils import generate_bentoml_deployment_snapshot_path

SERVERLESS_PROVIDER = {
    'aws-lambda': 'aws-python3',
    'aws-lambda-py2': 'aws-python',
    'gcp-function': 'google-python',
}


def generate_base_serverless_files(output_path, platform, name):
    subprocess.call(['serverless', 'create', '--template', platform, '--name', name],
                    cwd=output_path)
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


def check_serverless_compatiable_version():
    if which('serverless') is None:
        raise ValueError('Serverless framework is not installed, please visit ' +
                         'www.serverless.com for install instructions.')

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

    generate_base_serverless_files(output_path, provider, bento_service.name)

    if provider == 'google-python':
        update_gcp_function_configuration(bento_service, output_path, additional_options)
    elif provider == 'aws-lambda' or provider == 'aws-lambda-py3':
        update_aws_lambda_configuration(bento_service, output_path, additional_options)
    else:
        raise BentoMLException(("{provider} is not supported in current version of BentoML",
                                 provider))

    shutil.copy(os.path.join(archive_path, 'requirements.txt'), output_path)
    add_model_service_archive(bento_service, archive_path, output_path)

    return os.path.realpath(output_path)


def deploy_with_serverless(platform, archive_path, extra_args):
    bento_service = load(archive_path)
    output_path = generate_serverless_bundle(bento_service, platform, archive_path, extra_args)
    deploy_serverless_file(output_path)
    return output_path
