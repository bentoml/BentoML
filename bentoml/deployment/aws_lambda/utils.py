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
import tarfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from bentoml.exceptions import BentoMLException
from bentoml.utils.s3 import upload_directory_to_s3

logger = logging.getLogger(__name__)

# Maximum size for Lambda function bundle 250MB, leaving 1mb offset
LAMBDA_FUNCTION_LIMIT = 249000000
# Total disk size that user has access to.  Lambda function bundle + /tmp 512MB, leaving
# 1mb offset
LAMBDA_TEMPORARY_DIRECTORY_MAX_LIMIT = 511000000
LAMBDA_FUNCTION_MAX_LIMIT = LAMBDA_FUNCTION_LIMIT + LAMBDA_TEMPORARY_DIRECTORY_MAX_LIMIT


def ensure_sam_available_or_raise():
    try:
        import samcli

        if samcli.__version__ != '0.33.1':
            raise BentoMLException(
                'aws-sam-cli package requires version 0.33.1 '
                'Install the package with `pip install -U aws-sam-cli==0.33.1`'
            )
    except ImportError:
        raise MissingDependencyException(
            'aws-sam-cli package is required. Install '
            'with `pip install --user aws-sam-cli`'
        )


def cleanup_build_files(project_dir, api_name):
    build_dir = os.path.join(project_dir, '.aws-sam/build/{}'.format(api_name))
    logger.debug('Cleaning up unused files in SAM built directory %s', build_dir)
    for root, dirs, files in os.walk(build_dir):
        if 'tests' in dirs:
            logger.debug('removing dir: %s', os.path.join(root, 'tests'))
            shutil.rmtree(os.path.join(root, 'tests'))
        if 'test' in dirs:
            logger.debug('removing dir: %s', os.path.join(root, 'test'))
            shutil.rmtree(os.path.join(root, 'test'))
        if 'examples' in dirs:
            logger.debug('removing dir: %s', os.path.join(root, 'examples'))
            shutil.rmtree(os.path.join(root, 'examples'))
        if '__pycache__' in dirs:
            logger.debug('removing dir: %s', os.path.join(root, '__pycache__'))
            shutil.rmtree(os.path.join(root, '__pycache__'))

        for dir_name in filter(lambda x: re.match('^.*\\.dist-info$', x), dirs):
            logger.debug('removing dir: %s', dir_name)
            shutil.rmtree(os.path.join(root, dir_name))

        for file in filter(lambda x: re.match('^.*\\.egg-info$', x), files):
            logger.debug('removing file: %s', os.path.join(root, file))
            os.remove(os.path.join(root, file))

        for file in filter(lambda x: re.match('^.*\\.pyc$', x), files):
            logger.debug('removing file: %s', os.path.join(root, file))
            os.remove(os.path.join(root, file))

        if 'wheel' in files:
            logger.debug('removing file: %s', os.path.join(root, 'wheel'))
            os.remove(os.path.join(root, 'wheel'))
        if 'pip' in files:
            logger.debug('removing file: %s', os.path.join(root, 'pip'))
            os.remove(os.path.join(root, 'pip'))
        if 'caff2' in files:
            logger.debug('removing file: %s', os.path.join(root, 'caff2'))
            os.remove(os.path.join(root, 'caff2'))


def call_sam_command(command, project_dir):
    command = ['sam'] + command
    proc = subprocess.Popen(
        command, cwd=project_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = proc.communicate()
    logger.debug('SAM cmd %s output: %s', command, stdout.decode('utf-8'))
    return proc.returncode, stdout.decode('utf-8'), stderr.decode('utf-8')


def lambda_package(project_dir, aws_region, s3_bucket_name, deployment_prefix):
    prefix_path = os.path.join(deployment_prefix, 'lambda-functions')
    build_dir = os.path.join(project_dir, '.aws-sam', 'build')

    return_code, stdout, stderr = call_sam_command(
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
            '--region',
            aws_region,
        ],
        build_dir,
    )
    if return_code != 0:
        error_message = stderr
        if not error_message:
            error_message = stdout
        raise BentoMLException(
            'Failed to package lambda function. {}'.format(error_message)
        )
    else:
        return stdout


def lambda_deploy(project_dir, aws_region, stack_name):
    template_file = os.path.join(project_dir, '.aws-sam', 'build', 'packaged.yaml')
    return_code, stdout, stderr = call_sam_command(
        [
            'deploy',
            '--stack-name',
            stack_name,
            '--capabilities',
            'CAPABILITY_IAM',
            '--template-file',
            template_file,
            '--region',
            aws_region,
        ],
        project_dir,
    )
    if return_code != 0:
        error_message = stderr
        if not error_message:
            error_message = stdout
        raise BentoMLException(
            'Failed to deploy lambda function. {}'.format(error_message)
        )
    else:
        return stdout


def validate_lambda_template(template_file, aws_region, sam_project_path):
    status_code, stdout, stderr = call_sam_command(
        ['validate', '--template-file', template_file, '--region', aws_region],
        sam_project_path,
    )
    if status_code != 0:
        error_message = stderr
        if not error_message:
            error_message = stdout
        raise BentoMLException(
            'Failed to validate lambda template. {}'.format(error_message)
        )


def init_sam_project(
    sam_project_path,
    bento_service_bundle_path,
    deployment_name,
    bento_name,
    api_names,
    aws_region,
):
    function_path = os.path.join(sam_project_path, deployment_name)
    os.mkdir(function_path)
    # Copy requirements.txt
    logger.debug('Coping requirements.txt')
    requirement_txt_path = os.path.join(bento_service_bundle_path, 'requirements.txt')
    shutil.copy(requirement_txt_path, function_path)

    bundled_dep_path = os.path.join(
        bento_service_bundle_path, 'bundled_pip_dependencies'
    )
    if os.path.isdir(bundled_dep_path):
        # Copy bundled pip dependencies
        logger.debug('Coping bundled_dependencies')
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

    logger.debug('Creating python files for lambda function')
    # Create empty __init__.py
    open(os.path.join(function_path, '__init__.py'), 'w').close()

    app_py_path = os.path.join(os.path.dirname(__file__), 'lambda_app.py')
    shutil.copy(app_py_path, os.path.join(function_path, 'app.py'))
    unzip_requirement_py_path = os.path.join(
        os.path.dirname(__file__), 'unzip_requirements.py'
    )
    shutil.copy(
        unzip_requirement_py_path, os.path.join(function_path, 'unzip_requirements.py')
    )

    logger.info('Building lambda project')
    return_code, stdout, stderr = call_sam_command(
        ['build', '--use-container', '--region', aws_region], sam_project_path
    )
    if return_code != 0:
        error_message = stderr
        if not error_message:
            error_message = stdout
        raise BentoMLException(
            'Failed to build lambda function. {}'.format(error_message)
        )
    logger.debug('Removing unnecessary files to free up space')
    for api_name in api_names:
        cleanup_build_files(sam_project_path, api_name)


def _sum_total_directory_size(directory):
    dir_size = sum(
        f.stat().st_size for f in Path(directory).glob('**/*') if f.is_file()
    )
    return dir_size


def is_build_function_size_under_lambda_limit(build_directory):
    dir_size = _sum_total_directory_size(build_directory)
    logger.debug('Directory %s size is %d', build_directory, dir_size)
    return dir_size <= LAMBDA_FUNCTION_LIMIT


def is_build_function_size_over_max_lambda_size(build_directory):
    dir_size = _sum_total_directory_size(build_directory)
    return dir_size > LAMBDA_FUNCTION_MAX_LIMIT


def reduce_lambda_build_directory_size(
    build_directory,
    region,
    s3_bucket,
    deployment_prefix,
    function_name,
    lambda_project_dir,
    bento_service_name,
    s3_artifacts_prefix,
):
    additional_requirements_path = os.path.join(
        lambda_project_dir, 'additional_requirements', function_name
    )
    upload_dir_path = os.path.join(additional_requirements_path, 'requirements')
    s3_client = boto3.client('s3', region)

    # list of modules that HAVE to be part of the function bundle
    lambda_function_required_bundled_list = [
        'botocore',
        'boto3',
        'bentoml',
        'app.py',
        '__init__.py',
        'unzip_requirements.py',
        bento_service_name,
    ]
    current_function_size = sum(
        _sum_total_directory_size(os.path.join(build_directory, i))
        for i in lambda_function_required_bundled_list
    )
    logger.debug('Basic function bundle size is %d', current_function_size)
    if current_function_size > LAMBDA_FUNCTION_LIMIT:
        # Current function bundle is still over the limit. If bento bundle has
        # "artifacts" directory, we will move artifacts to s3 to reduce function bundle
        # size. If there is no artifacts directory, we will use build-in boto3/botocore
        # from lambda function, instead of bundle our own.
        # We are making assumption of bento bundle is relatively small.
        artifact_directory = os.path.join(
            build_directory, bento_service_name, 'artifacts'
        )
        if os.path.exists(artifact_directory):
            logger.debug(
                'The function bundle is over limit. Moving artifacts to s3 bucket.'
            )
            upload_directory_to_s3(
                upload_directory_path=artifact_directory,
                region=region,
                bucket_name=s3_bucket,
                s3_path_prefix=s3_artifacts_prefix,
            )
            shutil.rmtree(artifact_directory)
        else:
            # we are just going to use the boto3/botocore from the lambda runtime,
            # instead of packing our own
            shutil.rmtree(os.path.join(build_directory, 'boto3'))
            shutil.rmtree(os.path.join(build_directory, 'botocore'))
            logger.debug(
                'Remove "boto3" and "botocore" from function bundle, using Lambda '
                'provide modules instead'
            )
        current_function_size = sum(
            _sum_total_directory_size(os.path.join(build_directory, i))
            for i in lambda_function_required_bundled_list
        )
        logger.debug('The new function bundle size is %d', current_function_size)

    # Continue include modules until the function bundle size is at the limit. Rest of
    # the modules will be tar and upload to s3 bucket
    for file_or_dir_name in os.listdir(build_directory):
        if file_or_dir_name in lambda_function_required_bundled_list:
            logger.debug(
                'Including "%s" it is part of required bundle.', file_or_dir_name
            )
            continue
        else:
            function_size_limit = current_function_size + _sum_total_directory_size(
                os.path.join(build_directory, file_or_dir_name)
            )
            if function_size_limit >= LAMBDA_FUNCTION_LIMIT:
                logger.debug(
                    'Lambda function size %d is over the limit. Moving item "%s" '
                    'out of the bundle directory',
                    function_size_limit,
                    file_or_dir_name,
                )
                package_path = os.path.join(build_directory, file_or_dir_name)
                copy_dst_path = os.path.join(upload_dir_path, file_or_dir_name)
                if os.path.isdir(package_path):
                    shutil.copytree(package_path, copy_dst_path)
                    shutil.rmtree(package_path)
                else:
                    shutil.copyfile(package_path, copy_dst_path)
                    os.remove(package_path)
            else:
                logger.debug(
                    'Including "%s" Now the current lambda function size is %d',
                    file_or_dir_name,
                    function_size_limit,
                )
                current_function_size = function_size_limit
    logger.debug('Final Lambda function bundle size is %d', current_function_size)
    additional_requirements_size = _sum_total_directory_size(
        additional_requirements_path
    )
    logger.debug('Additional requirement size is %d', additional_requirements_size)
    logger.debug('zip up additional requirement packages')
    tar_file_path = os.path.join(additional_requirements_path, 'requirements.tar')
    with tarfile.open(tar_file_path, 'w:gz') as tar:
        tar.add(upload_dir_path, arcname='requirements')
    logger.debug(
        'Uploading requirements.tar to %s/%s', s3_bucket, deployment_prefix
    )
    try:
        s3_client.upload_file(
            tar_file_path,
            s3_bucket,
            os.path.join(deployment_prefix, 'requirements.tar'),
        )
    except ClientError as e:
        raise BentoMLException(str(e))
