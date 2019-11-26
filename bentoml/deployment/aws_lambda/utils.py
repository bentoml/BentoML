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
import re

import boto3

from bentoml.exceptions import BentoMLException, BentoMLDeploymentException

logger = logging.getLogger(__name__)


def ensure_sam_available_or_raise():
    try:
        import samcli  # noqa
    except ImportError:
        raise ImportError(
            'aws-sam-cli package is required. Install '
            'with `pip install --user aws-sam-cli`'
        )


def cleanup_build_files(project_dir, api_name):
    build_dir = os.path.join(project_dir, '.aws-sam/build/{}'.format(api_name))
    logger.debug('Cleaning up unused files in SAM built directory {}'.format(build_dir))
    try:
        for root, dirs, files in os.walk(build_dir):
            if 'tests' in dirs:
                logger.debug('removing dir: ' + os.path.join(root, 'tests'))
                shutil.rmtree(os.path.join(root, 'tests'))
            if 'test' in dirs:
                logger.debug('removing dir: ' + os.path.join(root, 'test'))
                shutil.rmtree(os.path.join(root, 'test'))
            if 'examples' in dirs:
                logger.debug('removing dir: ' + os.path.join(root, 'examples'))
                shutil.rmtree(os.path.join(root, 'examples'))
            if '__pycache__' in dirs:
                logger.debug('removing dir: ' + os.path.join(root, '__pycache__'))
                shutil.rmtree(os.path.join(root, '__pycache__'))

            for dir_name in filter(lambda x: re.match('^.*\\.dist-info$', x), dirs):
                logger.debug('removing dir ' + dir_name)
                shutil.rmtree(os.path.join(root, dir_name))

            for file in filter(lambda x: re.match('^.*\\.egg-info$', x), files):
                logger.debug('removing file: ' + os.path.join(root, file))
                os.remove(os.path.join(root, file))

            for file in filter(lambda x: re.match('^.*\\.pyc$', x), files):
                logger.debug('removing file: ' + os.path.join(root, file))
                os.remove(os.path.join(root, file))

            if 'caff2' in files:
                logger.debug('removing file: ' + os.path.join(root, 'caff2'))
                os.remove(os.path.join(root, 'caff2'))
            if 'wheel' in files:
                logger.debug('removing file: ' + os.path.join(root, 'wheel'))
                os.remove(os.path.join(root, 'wheel'))
            if 'pip' in files:
                logger.debug('removing file: ' + os.path.join(root, 'pip'))
                os.remove(os.path.join(root, 'pip'))
    except Exception as e:
        logger.error(str(e))
        return


def call_sam_command(command, project_dir):
    command = ['sam'] + command
    with subprocess.Popen(
        command, cwd=project_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as proc:
        stdout = proc.stdout.read().decode('utf-8')
        logger.debug(
            'SAM cmd {command} output: {output}'.format(command=command, output=stdout)
        )
    return stdout


def lambda_package(project_dir, s3_bucket_name, deployment_prefix):
    prefix_path = os.path.join(deployment_prefix, 'lambda-functions')
    build_dir = os.path.join(project_dir, '.aws-sam', 'build')

    call_sam_command(
        [
            'package',
            '--force-upload',
            '--s3-bucket',
            s3_bucket_name,
            '--s3-prefix',
            prefix_path,
            '--template-file',
            'template.yaml',
            '--output-template-file',
            'packaged.yaml',
        ],
        build_dir,
    )


def lambda_deploy(project_dir, stack_name):
    template_file = os.path.join(project_dir, '.aws-sam', 'build', 'packaged.yaml')
    try:
        from samcli.lib.samlib.cloudformation_command import execute_command
    except ImportError:
        raise ImportError(
            'aws-sam-cli package is required. Install '
            'with `pip install --user aws-sam-cli`'
        )

    execute_command(
        "deploy",
        ["--stack-name", stack_name, "--capabilities", "CAPABILITY_IAM"],
        template_file=template_file,
    )


def validate_lambda_template(template_file):
    try:
        from samtranslator.translator.managed_policy_translator import (
            ManagedPolicyLoader,
        )
        from botocore.exceptions import NoCredentialsError
        from samcli.commands.validate.lib.exceptions import InvalidSamDocumentException
        from samcli.commands.validate.lib.sam_template_validator import (
            SamTemplateValidator,
        )
        from samcli.commands.validate.validate import _read_sam_file
    except ImportError:
        raise ImportError(
            'aws-sam-cli package is required. Install '
            'with `pip install --user aws-sam-cli`'
        )

    sam_template = _read_sam_file(template_file)
    iam_client = boto3.client("iam")
    validator = SamTemplateValidator(sam_template, ManagedPolicyLoader(iam_client))
    try:
        validator.is_valid()
    except InvalidSamDocumentException as e:
        raise BentoMLDeploymentException('Invalid SAM Template for AWS Lambda.')
    except NoCredentialsError as e:
        raise BentoMLDeploymentException(
            'AWS Credential are required, please configure your credentials.'
        )


def init_sam_project(
    sam_project_path, bento_service_bundle_path, deployment_name, bento_name, api_names
):
    function_path = os.path.join(sam_project_path, deployment_name)
    os.mkdir(function_path)
    # Copy requirements.txt
    logger.debug('Coping requirements.txt')
    requirement_txt_path = os.path.join(bento_service_bundle_path, 'requirements.txt')
    shutil.copy(requirement_txt_path, function_path)

    # Copy bundled pip dependencies
    logger.debug('Coping bundled_dependencies')
    bundled_dep_path = os.path.join(
        bento_service_bundle_path, 'bundled_pip_dependencies'
    )
    if os.path.isdir(bundled_dep_path):
        shutil.copytree(
            bundled_dep_path, os.path.join(function_path, 'bundled_pip_dependencies')
        )
        bundled_files = os.listdir(
            os.path.join(function_path, 'bundled_pip_dependencies')
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

        logger.debug('Updating requirements.txt')
        with open(
            os.path.join(function_path, 'requirements.txt'), 'r+'
        ) as requirement_file:
            required_modules = requirement_file.readlines()
            if has_bentoml_bundle:
                # Assuming bentoml is always the first one in requirements.txt.
                # We are removing it
                required_modules = required_modules[1:]
            required_modules = required_modules + bundled_files
            # Write from beginning of the file, instead of appending to the end.
            requirement_file.seek(0)
            requirement_file.writelines(required_modules)

    # Copy bento_service_model
    logger.debug('Coping model directory')
    model_path = os.path.join(bento_service_bundle_path, bento_name)
    shutil.copytree(model_path, os.path.join(function_path, bento_name))

    # remove the artifacts dir. Artifacts already uploaded to s3
    logger.debug('Removing artifacts directory')
    shutil.rmtree(os.path.join(function_path, bento_name, 'artifacts'))

    logger.debug('Creating python files for lambda function')
    # Create empty __init__.py
    open(os.path.join(function_path, '__init__.py'), 'w').close()

    app_py_path = os.path.join(os.path.dirname(__file__), 'lambda_app.py')
    shutil.copy('/'.join(app_py_path), os.path.join(function_path, 'app.py'))

    logger.info('Building lambda project')
    build_result = call_sam_command(['build', '--use-container'], sam_project_path)
    if 'Build Failed' in build_result:
        raise BentoMLException('Build Lambda project failed')
    logger.debug('Removing unnecessary files to free up space')
    for api_name in api_names:
        cleanup_build_files(sam_project_path, api_name)
