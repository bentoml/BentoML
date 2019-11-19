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
from botocore.exceptions import ClientError

from bentoml.exceptions import BentoMLException, BentoMLMissingDependencyException


logger = logging.getLogger(__name__)


AWS_LAMBDA_APP_PY_TEMPLATE_HEADER = """\
import os
import boto3

# Set BENTOML_HOME to /tmp directory due to AWS lambda disk access restrictions
os.environ['BENTOML_HOME'] = '/tmp/bentoml/'
from bentoml.bundler import load_bento_service_class

file_dir = os.path.join('/tmp/bentoml/', 'artifacts')
os.mkdir(file_dir)

# Download artifacts from s3 to '/tmp/bentoml/artifacts' dir
s3_client = boto3.client('s3')
s3_bucket = '{s3_bucket}'
artifacts_prefix = '{artifacts_prefix}'
try:
    list_content_result = s3_client.list_objects(
        Bucket=s3_bucket,
        Prefix=artifacts_prefix
    )
    for content in list_content_result['Contents']:
        file_name = content['Key'].split('/')[-1]
        file_path = os.path.join(file_dir, file_name)
        if not os.path.isfile(file_path):
            s3_client.download_file(s3_bucket, content['Key'], file_path)
        else:
            print('File {{}} already exists'.format(file_path))
except Exception as e:
    print('Error getting object from bucket {s3_bucket}, {{}}'.format(e))
    raise e

bento_service_cls = load_bento_service_class(bundle_path='./{bento_name}')
# Set _bento_service_bundle_path to None, so it won't automatically load artifacts when
# we init an instance
bento_service_cls._bento_service_bundle_path = None
bento_service = bento_service_cls()
bento_service._load_artifacts('/tmp/bentoml')

"""

AWS_FUNCTION_TEMPLATE = """\
def {api_name}(event, context):
    api = bento_service.get_service_api('{api_name}')

    return api.handle_aws_lambda_event(event)

"""


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
    logger.debug('Cleaning up for dir {}'.format(build_dir))
    try:
        for root, dirs, files in os.walk(build_dir):
            if 'tests' in dirs:
                logger.debug('removing dir: ' + os.path.join(root, 'tests'))
                shutil.rmtree(os.path.join(root, 'tests'))
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
            if 'libtorch.so' in files:
                logger.debug('removing file: ' + os.path.join(root, 'libtorch.so'))
                os.remove(os.path.join(root, 'libtorch.so'))
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
    build_dir = os.path.join(project_dir, '.aws-sam', 'build')
    call_sam_command(
        [
            'deploy',
            '--template-file',
            'packaged.yaml',
            '--stack-name',
            stack_name,
            # This option is not in help listing. without it, cloud formation will
            # result in review_in_progress
            # https://github.com/awslabs/serverless-application-model/issues/51
            '--capabilities',
            'CAPABILITY_IAM',
        ],
        build_dir,
    )


def create_s3_bucket_if_not_exists(bucket_name, region):
    s3_client = boto3.client('s3', region)
    try:
        s3_client.get_bucket_acl(Bucket=bucket_name)
        logger.debug('Use existing s3 bucket')
    except ClientError as error:
        if error.response and error.response['Error']['Code'] == 'NoSuchBucket':
            logger.debug(
                'S3 bucket {} does not exist. Creating it now'.format(bucket_name)
            )
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region},
            )
        else:
            raise error


def check_s3_bucket_exists_or_raise(bucket_name, region):
    s3_client = boto3.client('s3', region)
    try:
        s3_client.get_bucket_acl(Bucket=bucket_name)
    except ClientError as error:
        if error.response and error.response['Error']['Code'] == 'NoSuchBucket':
            raise BentoMLException(
                'S3 bucket {} does not exist. '
                'Please create it and try again'.format(bucket_name)
            )
        else:
            raise error

    pass


def upload_artifacts_to_s3(
    region, bucket_name, path_prefix, bento_archive_path, bento_name
):
    artifacts_path = os.path.join(bento_archive_path, bento_name, 'artifacts')
    s3_client = boto3.client('s3', region)
    try:
        for file_name in os.listdir(artifacts_path):
            if file_name != '__init__.py':
                logger.debug(
                    'Uploading {name} to s3 {location}'.format(
                        name=file_name, location=bucket_name + '/' + path_prefix
                    )
                )
                s3_client.upload_file(
                    os.path.join(artifacts_path, file_name),
                    bucket_name,
                    '{prefix}/{name}'.format(prefix=path_prefix, name=file_name),
                )
    except Exception as error:
        raise BentoMLException(str(error))


def generate_aws_lambda_app_py(
    function_path, s3_bucket, artifacts_prefix, bento_name, api_names
):
    with open(os.path.join(function_path, 'app.py'), 'w') as f:
        f.write(
            AWS_LAMBDA_APP_PY_TEMPLATE_HEADER.format(
                bento_name=bento_name,
                s3_bucket=s3_bucket,
                artifacts_prefix=artifacts_prefix,
            )
        )
        for api_name in api_names:
            api_content = AWS_FUNCTION_TEMPLATE.format(api_name=api_name)
            f.write(api_content)


def init_sam_project(
    sam_project_path,
    bento_archive_path,
    deployment_name,
    bento_name,
    api_names,
    s3_bucket,
    artifacts_prefix,
):
    function_path = os.path.join(sam_project_path, deployment_name)
    os.mkdir(function_path)
    # Copy requirements.txt
    logger.debug('Coping requirements.txt')
    requirement_txt_path = os.path.join(bento_archive_path, 'requirements.txt')
    shutil.copy(requirement_txt_path, function_path)

    # Copy bundled pip dependencies
    logger.debug('Coping bundled_dependencies')
    bundled_dep_path = os.path.join(bento_archive_path, 'bundled_pip_dependencies')
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
    model_path = os.path.join(bento_archive_path, bento_name)
    shutil.copytree(model_path, os.path.join(function_path, bento_name))

    # remove the artifacts dir. Artifacts already uploaded to s3
    logger.debug('Removing artifacts directory')
    shutil.rmtree(os.path.join(function_path, bento_name, 'artifacts'))

    # Create empty __init__.py
    open(os.path.join(function_path, '__init__.py'), 'w').close()
    logger.debug('Creating app.py for lambda function')
    generate_aws_lambda_app_py(
        function_path, s3_bucket, artifacts_prefix, bento_name, api_names
    )

    logger.info('Building lambda project')
    build_result = call_sam_command(['build', '--use-container'], sam_project_path)
    if 'Build Failed' in build_result:
        raise BentoMLException('Build Lambda project failed')
    logger.debug('Removing unnecessary files to free up space')
    for api_name in api_names:
        cleanup_build_files(sam_project_path, api_name)
