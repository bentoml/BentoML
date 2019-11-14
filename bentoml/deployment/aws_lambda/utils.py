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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import subprocess
import logging

import boto3

from bentoml.exceptions import BentoMLException, BentoMLMissingDependencyException

UNZIP_REQUIREMENTS_PY = """\
import os
import shutil
import sys
import zipfile


pkgdir = '/tmp/sls-py-req'

sys.path.append(pkgdir)

if not os.path.exists(pkgdir):
    tempdir = '/tmp/_temp-sls-py-req'
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)

    default_layer_root = '/opt'
    lambda_root = os.getcwd() if os.environ.get('IS_LOCAL') == 'true' else default_layer_root
    zip_requirements = os.path.join(lambda_root, '.requirements.zip')

    zipfile.ZipFile(zip_requirements, 'r').extractall(tempdir)
    os.rename(tempdir, pkgdir)  # Atomic
"""

logger = logging.getLogger(__name__)


def ensure_sam_available_or_raise():
    # for FileNotFoundError doesn't exist in py2.7. check_output raise OSError instead
    import six

    if six.PY3:
        not_found_error = FileNotFoundError
    else:
        not_found_error = OSError

    try:
        subprocess.check_output(['sam', '--version'])
    except subprocess.CalledProcessError as error:
        raise BentoMLException('Error executing sam command: {}'.format(error.output))
    except not_found_error:
        raise BentoMLMissingDependencyException(
            'SAM is required for AWS Lambda deployment. Please visit '
            'https://aws.amazon.com/serverless/sam for instructions'
        )


def cleanup_build_files(project_dir, api_name):
    build_dir = os.path.join(project_dir, '.aws-sam/build/{}'.format(api_name))
    subprocess.check_output(['rm', '-rf', './{*.egg-info,*.dist-info}'])


def call_sam_command(command, project_dir):
    command = ['sam'] + command
    with subprocess.Popen(
        command, cwd=project_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        stdout = proc.stdout.read().decode('utf-8')
        logger.debug('SAM cmd output: %s', stdout)
    return stdout


def lambda_build(project_dir, api_names):
    call_sam_command(['build', '-u'], project_dir)
    for api_name in api_names:
        cleanup_build_files(project_dir, api_name)


def lambda_package(project_dir, s3_bucket_name):
    call_sam_command(
        [
            'package',
            '--force-upload',
            '--s3-bucket',
            s3_bucket_name,
            '--template-file',
            'template.yaml',
        ],
        project_dir,
    )


def lambda_deploy(project_dir, stack_name):
    call_sam_command(
        ['deploy', '--template-file', 'template.yaml', '--stack-name', stack_name],
        project_dir,
    )


def upload_artifacts_to_s3(region, bucket_name, bento_archive_path, bento_name):
    artifacts_path = os.path.join(bento_archive_path, bento_name, 'artifacts')
    s3_client = boto3.client('s3', region)

    try:
        for file_name in os.listdir(artifacts_path):
            s3_client.upload_file(
                os.path.join(artifacts_path, file_name),
                bucket_name,
                '/artifacts/{}'.format(file_name),
            )
    except Exception as error:
        raise BentoMLException(str(error))


def create_dependency_layer(sam_project_path, s3_bucket_name):
    pass


def init_sam_project(sam_project_path, bento_archive_path, bento_name, api_names):
    # Copy requirements.txt
    requirement_txt_path = os.path.join(bento_archive_path, 'requirements.txt')
    shutil.copy(requirement_txt_path, sam_project_path)

    # Copy bundled pip dependencies
    bundled_dep_path = os.path.join(bento_archive_path, 'bundled_pip_dependencies')
    if os.path.isdir(bundled_dep_path):
        shutil.copytree(
            bundled_dep_path, os.path.join(sam_project_path, 'bundled_pip_dependencies')
        )
        bundled_files = os.listdir(
            os.path.join(sam_project_path, 'bundled_pip_dependencies')
        )
        has_bentoml_bundle = False
        for index, bundled_file_name in enumerate(bundled_files):
            bundled_files[index] = '\n./bundled_pip_dependencies/{}'.format(
                bundled_file_name
            )
            # If file name start with `BentoML-`, assuming it is a
            # bentoml targz bundle
            if bundled_file_name.startswith('BentoML-'):
                has_bentoml_bundle = True

        with open(
            os.path.join(sam_project_path, 'requirements.txt'), 'r+'
        ) as requirement_file:
            required_modules = requirement_file.readlines()
            if has_bentoml_bundle:
                # Assuming bentoml is always the first one in
                # requirements.txt. We are removing it
                required_modules = required_modules[1:]
            required_modules = required_modules + bundled_files
            # Write from beginning of the file, instead of appending to
            # the end.
            requirement_file.seek(0)
            requirement_file.writelines(required_modules)

    lambda_build()
    cleanup_build_files()

    # Copy bento_service_model
    model_path = os.path.join(bento_archive_path, bento_name)
    shutil.copytree(model_path, os.path.join(sam_project_path, bento_name))

    # remove the artifacts dir. Artifacts will be upload to S3
    shutil.rmtree(os.path.join(sam_project_path, bento_name, 'artifacts'))

    # generate app.py
    with open(os.path.join(sam_project_path, 'app.py'), 'w') as f:
        f.write('')
        for api_name in api_names:
            api_content = ''
            f.write(api_content)

    # generate unzip_requirements.py
    with open(os.path.join(sam_project_path, 'unzip_requirements.py'), 'w') as f:
        f.write(UNZIP_REQUIREMENTS_PY)

