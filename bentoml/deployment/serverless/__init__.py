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
from subprocess import PIPE

from ruamel.yaml import YAML
from packaging import version

from bentoml.archive import load, load_bentoml_config
from bentoml.utils import Path
from bentoml.utils.tempdir import TempDirectory
from bentoml.utils.whichcraft import which
from bentoml.utils.exceptions import BentoMLException
from bentoml.deployment.serverless.aws_lambda_template import create_aws_lambda_bundle, \
    DEFAULT_AWS_DEPLOY_STAGE, DEFAULT_AWS_REGION
from bentoml.deployment.serverless.gcp_function_template import create_gcp_function_bundle, \
    DEFAULT_GCP_REGION, DEFAULT_GCP_DEPLOY_STAGE
from bentoml.deployment.utils import generate_bentoml_deployment_snapshot_path


SERVERLESS_PROVIDER = {
    'aws-lambda': 'aws-python3',
    'aws-lambda-py2': 'aws-python',
    'gcp-function': 'google-python',
}

DEFAULT_GCP_REGION = 'us-west2'


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


def check_serverless_deployment_status(platform, archive_path, region, stage):
    check_serverless_compatiable_version()

    service_name = load_bentoml_config(archive_path)['metadata']['service_name']
    if platform == 'google-python':
        config = {
            "service": service_name,
            "provider": {
                "name": "google",
                "region": DEFAULT_GCP_REGION if region is None else region,
                "stage": DEFAULT_GCP_DEPLOY_STAGE if stage is None else stage
            }
        }
    elif platform == 'aws-lambda' or platform == 'aws-lambda-py2':
        config = {
            "service": service_name,
            "provider": {
                "name": "aws",
                "region": DEFAULT_AWS_REGION if region is None else region,
                "stage": DEFAULT_AWS_DEPLOY_STAGE if stage is None else stage
            }
        }
    else:
        raise BentoMLException(
            'check serverless does not support platform %s at the moment' % platform)
    yaml = YAML()
    with TempDirectory() as tempdir:
        saved_path = os.path.join(tempdir, 'serverless.yml')
        yaml.dump(config, Path(saved_path))
        with subprocess.Popen(['serverless', 'info'], cwd=tempdir, stdout=PIPE,
                              stderr=PIPE) as proc:
            content = proc.stdout.read()
            if 'Serverless Error' in content:
                return False
            else:
                return True


class ServerlessDeployment(object):
    """Managing deployment operations for serverless
    """

    def __init__(self, platform, archive_path, region, stage):
        self.platform = platform
        self.provider = SERVERLESS_PROVIDER[platform]
        self.archive_path = archive_path
        self.bento_service = load(archive_path)
        if platform == 'google-python':
            self.region = DEFAULT_GCP_REGION if region is None else region
            self.stage = DEFAULT_GCP_DEPLOY_STAGE if stage is None else stage
        elif platform == 'aws-lambda' or platform == 'aws-lambda-py':
            self.region = DEFAULT_AWS_REGION if region is None else region
            self.stage = DEFAULT_AWS_DEPLOY_STAGE if stage is None else stage

    def generate_bundle(self):
        output_path = generate_bentoml_deployment_snapshot_path(self.bento_service.name, self.platform)
        Path(output_path).mkdir(parents=True, exist_ok=False)

        # Calling serverless command to generate templated project
        subprocess.call(['serverless', 'create', '--template', self.provider, '--name', self.bento_service.name],
                        cwd=output_path)
        if self.platform == 'google-python':
            create_gcp_function_bundle(self.bento_service, output_path, self.region, self.stage)
        elif self.platform == 'aws-lambda' or self.platform == 'aws-lambda-py2':
            # Installing two additional plugins to make it works for AWS lambda
            # serverless-python-requirements will packaging required python modules, and automatically
            # compress and create layer
            subprocess.call(['serverless', 'plugin', 'install', '-n', 'serverless-python-requirements'],
                            cwd=output_path)
            subprocess.call(['serverless', 'plugin', 'install', '-n', 'serverless-apigw-binary'],
                            cwd=output_path)
            create_aws_lambda_bundle(self.bento_service, output_path, self.region, self.stage)
        else:
            raise BentoMLException("%s is not supported in current version of BentoML" % self.provider)

        shutil.copy(os.path.join(self.archive_path, 'requirements.txt'), output_path)

        model_serivce_archive_path = os.path.join(output_path, self.bento_service.name)
        shutil.copytree(self.archive_path, model_serivce_archive_path)

        return os.path.realpath(output_path)

    def deploy(self):
        check_serverless_compatiable_version()

        output_path = self.generate_bundle()
        with subprocess.Popen(['serverless', 'deploy'], cwd=output_path, stdout=PIPE, stderr=PIPE) as proc:
            content_out = proc.stdout.read()
            if 'Serverless Error' in content_out:
                raise BentoMLException('Serverless Error')
            else:
                return output_path

    def check_status(self):
        check_serverless_compatiable_version()

        service_name = load_bentoml_config(self.archive_path)['metadata']['service_name']
        if self.platform == 'google-python':
            provider_name = 'google'
        elif self.platform == 'aws-lambda' or self.platform == 'aws-lambda-py2':
            provider_name = 'aws'
        else:
            raise BentoMLException(
                'check serverless does not support platform %s at the moment' % self.platform)
        config = {
            "service": service_name,
            "provider": {
                "name": provider_name,
                "region": self.region,
                "stage": self.stage
            }
        }
        yaml = YAML()
        with TempDirectory() as tempdir:
            saved_path = os.path.join(tempdir, 'serverless.yml')
            yaml.dump(config, Path(saved_path))
            with subprocess.Popen(['serverless', 'info'], cwd=tempdir, stdout=PIPE,
                                stderr=PIPE) as proc:
                content = proc.stdout.read()
                if 'Serverless Error' in content:
                    return False
                else:
                    return True

    def delete(self):
        check_serverless_compatiable_version()

        service_name = load_bentoml_config(self.archive_path)['metadata']['service_name']
        if self.platform == 'google-python':
            provider_name = 'google'
        elif self.platform == 'aws-lambda' or self.platform == 'aws-lambda-py2':
            provider_name = 'aws'
        config = {
            "service": service_name,
            "provider": {
                "name": provider_name,
                "region": self.region,
                "stage": self.stage
            }
        }
        yaml = YAML()
        with TempDirectory() as tempdir:
            saved_path = os.path.join(tempdir, 'serverless.yml')
            yaml.dump(config, Path(saved_path))
            subprocess.call(['serverless', 'remove'], cwd=tempdir)
            with subprocess.Popen(['serverless', 'remove'], cwd=tempdir, stdout=PIPE, stderr=PIPE) as proc:
                content = proc.stdout.read()
                if 'Serverless Error' in content:
                    return False
                else:
                    return True
