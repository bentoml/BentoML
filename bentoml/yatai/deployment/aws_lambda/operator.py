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

import json
import os
import shutil
import uuid
import logging
from pathlib import Path

import boto3

from bentoml.utils.ruamel_yaml import YAML
from bentoml.exceptions import (
    BentoMLException,
    InvalidArgument,
    YataiDeploymentException,
)
from bentoml.saved_bundle import loader
from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.utils.s3 import create_s3_bucket_if_not_exists
from bentoml.utils.tempdir import TempDirectory
from bentoml.yatai.deployment.aws_utils import (
    validate_sam_template,
    FAILED_CLOUDFORMATION_STACK_STATUS,
    cleanup_s3_bucket_if_exist,
)
from bentoml.yatai.deployment.aws_lambda.utils import (
    init_sam_project,
    lambda_deploy,
    lambda_package,
    reduce_bundle_size_and_upload_extra_resources_to_s3,
    total_file_or_directory_size,
    LAMBDA_FUNCTION_LIMIT,
    LAMBDA_FUNCTION_MAX_LIMIT,
)

from bentoml.yatai.deployment.operator import DeploymentOperatorBase
from bentoml.yatai.deployment.utils import (
    raise_if_api_names_not_found_in_bento_service_metadata,
)
from bentoml.yatai.deployment.aws_utils import (
    generate_aws_compatible_string,
    get_default_aws_region,
    ensure_sam_available_or_raise,
)
from bentoml.yatai.deployment.docker_utils import ensure_docker_available_or_raise
from bentoml.yatai.proto import status_pb2
from bentoml.yatai.proto.deployment_pb2 import (
    DeploymentState,
    ApplyDeploymentResponse,
    DeleteDeploymentResponse,
    DescribeDeploymentResponse,
)
from bentoml.yatai.proto.repository_pb2 import GetBentoRequest, BentoUri
from bentoml.yatai.status import Status


logger = logging.getLogger(__name__)


def _create_aws_lambda_cloudformation_template_file(
    project_dir,
    namespace,
    deployment_name,
    deployment_path_prefix,
    api_names,
    bento_service_name,
    s3_bucket_name,
    py_runtime,
    memory_size,
    timeout,
):
    template_file_path = os.path.join(project_dir, 'template.yaml')
    yaml = YAML()
    sam_config = {
        'AWSTemplateFormatVersion': '2010-09-09',
        'Transform': 'AWS::Serverless-2016-10-31',
        'Globals': {
            'Function': {'Timeout': timeout, 'Runtime': py_runtime},
            'Api': {
                'BinaryMediaTypes': ['image~1*'],
                'Cors': "'*'",
                'Auth': {
                    'ApiKeyRequired': False,
                    'DefaultAuthorizer': 'NONE',
                    'AddDefaultAuthorizerToCorsPreflight': False,
                },
            },
        },
        'Resources': {},
        'Outputs': {
            'S3Bucket': {
                'Value': s3_bucket_name,
                'Description': 'S3 Bucket for saving artifacts and lambda bundle',
            }
        },
    }
    for api_name in api_names:
        sam_config['Resources'][api_name] = {
            'Type': 'AWS::Serverless::Function',
            'Properties': {
                'Runtime': py_runtime,
                'CodeUri': deployment_name + '/',
                'Handler': 'app.{}'.format(api_name),
                'FunctionName': f'{namespace}-{deployment_name}-{api_name}',
                'Timeout': timeout,
                'MemorySize': memory_size,
                'Events': {
                    'Api': {
                        'Type': 'Api',
                        'Properties': {
                            'Path': '/{}'.format(api_name),
                            'Method': 'post',
                        },
                    }
                },
                'Policies': [{'S3ReadPolicy': {'BucketName': s3_bucket_name}}],
                'Environment': {
                    'Variables': {
                        'BENTOML_BENTO_SERVICE_NAME': bento_service_name,
                        'BENTOML_API_NAME': api_name,
                        'BENTOML_S3_BUCKET': s3_bucket_name,
                        'BENTOML_DEPLOYMENT_PATH_PREFIX': deployment_path_prefix,
                    }
                },
            },
        }

    yaml.dump(sam_config, Path(template_file_path))

    # We add Outputs section separately, because the value should not
    # have "'" around !Sub
    with open(template_file_path, 'a') as f:
        f.write(
            """\
  EndpointUrl:
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.\
amazonaws.com/Prod"
    Description: URL for endpoint
"""
        )
    return template_file_path


def _deploy_lambda_function(
    deployment_pb,
    bento_service_metadata,
    deployment_spec,
    lambda_s3_bucket,
    lambda_deployment_config,
    bento_path,
):
    deployment_path_prefix = os.path.join(deployment_pb.namespace, deployment_pb.name)

    py_major, py_minor, _ = bento_service_metadata.env.python_version.split('.')
    if py_major != '3':
        raise BentoMLException('Python 2 is not supported for Lambda Deployment')
    python_runtime = 'python{}.{}'.format(py_major, py_minor)

    artifact_types = [item.artifact_type for item in bento_service_metadata.artifacts]
    if any(
        i in ['TensorflowSavedModelArtifact', 'KerasModelArtifact']
        for i in artifact_types
    ) and (py_major, py_minor) != ('3', '6'):
        raise BentoMLException(
            'AWS Lambda Deployment only supports BentoML services'
            'built with Python 3.6.x. To fix this, repack your'
            'service with the right Python version'
            '(hint: pyenv/anaconda) and try again'
        )

    api_names = (
        [lambda_deployment_config.api_name]
        if lambda_deployment_config.api_name
        else [api.name for api in bento_service_metadata.apis]
    )

    raise_if_api_names_not_found_in_bento_service_metadata(
        bento_service_metadata, api_names
    )

    with TempDirectory() as lambda_project_dir:
        logger.debug(
            'Generating cloudformation template.yaml for lambda project at %s',
            lambda_project_dir,
        )
        template_file_path = _create_aws_lambda_cloudformation_template_file(
            project_dir=lambda_project_dir,
            namespace=deployment_pb.namespace,
            deployment_name=deployment_pb.name,
            deployment_path_prefix=deployment_path_prefix,
            api_names=api_names,
            bento_service_name=deployment_spec.bento_name,
            s3_bucket_name=lambda_s3_bucket,
            py_runtime=python_runtime,
            memory_size=lambda_deployment_config.memory_size,
            timeout=lambda_deployment_config.timeout,
        )
        logger.debug('Validating generated template.yaml')
        validate_sam_template(
            template_file_path, lambda_deployment_config.region, lambda_project_dir,
        )
        logger.debug(
            'Initializing lambda project in directory: %s ...', lambda_project_dir,
        )
        init_sam_project(
            lambda_project_dir,
            bento_path,
            deployment_pb.name,
            deployment_spec.bento_name,
            api_names,
            aws_region=lambda_deployment_config.region,
        )
        for api_name in api_names:
            build_directory = os.path.join(
                lambda_project_dir, '.aws-sam', 'build', api_name
            )
            logger.debug(
                'Checking is function "%s" bundle under lambda size ' 'limit', api_name,
            )
            # Since we only use s3 get object in lambda function, and
            # lambda function pack their own boto3/botocore modules,
            # we will just delete those modules from function bundle
            # directory
            delete_list = ['boto3', 'botocore']
            for name in delete_list:
                logger.debug('Remove module "%s" from build directory', name)
                shutil.rmtree(os.path.join(build_directory, name))
            total_build_dir_size = total_file_or_directory_size(build_directory)
            if total_build_dir_size > LAMBDA_FUNCTION_MAX_LIMIT:
                raise BentoMLException(
                    'Build function size is over 700MB, max size '
                    'capable for AWS Lambda function'
                )
            if total_build_dir_size >= LAMBDA_FUNCTION_LIMIT:
                logger.debug(
                    'Function %s is over lambda size limit, attempting ' 'reduce it',
                    api_name,
                )
                reduce_bundle_size_and_upload_extra_resources_to_s3(
                    build_directory=build_directory,
                    region=lambda_deployment_config.region,
                    s3_bucket=lambda_s3_bucket,
                    deployment_prefix=deployment_path_prefix,
                    function_name=api_name,
                    lambda_project_dir=lambda_project_dir,
                )
            else:
                logger.debug(
                    'Function bundle is within Lambda limit, removing '
                    'download_extra_resources.py file from function bundle'
                )
                os.remove(os.path.join(build_directory, 'download_extra_resources.py'))
        logger.info('Packaging AWS Lambda project at %s ...', lambda_project_dir)
        lambda_package(
            lambda_project_dir,
            lambda_deployment_config.region,
            lambda_s3_bucket,
            deployment_path_prefix,
        )
        logger.info('Deploying lambda project')
        stack_name = generate_aws_compatible_string(
            deployment_pb.namespace + '-' + deployment_pb.name
        )
        lambda_deploy(
            lambda_project_dir, lambda_deployment_config.region, stack_name=stack_name,
        )


class AwsLambdaDeploymentOperator(DeploymentOperatorBase):
    def __init__(self, yatai_service):
        super(AwsLambdaDeploymentOperator, self).__init__(yatai_service)
        ensure_docker_available_or_raise()
        ensure_sam_available_or_raise()

    def add(self, deployment_pb):
        try:
            deployment_spec = deployment_pb.spec
            deployment_spec.aws_lambda_operator_config.region = (
                deployment_spec.aws_lambda_operator_config.region
                or get_default_aws_region()
            )
            if not deployment_spec.aws_lambda_operator_config.region:
                raise InvalidArgument('AWS region is missing')

            bento_pb = self.yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            if bento_pb.bento.uri.type not in (BentoUri.LOCAL, BentoUri.S3):
                raise BentoMLException(
                    'BentoML currently not support {} repository'.format(
                        BentoUri.StorageType.Name(bento_pb.bento.uri.type)
                    )
                )

            return self._add(deployment_pb, bento_pb, bento_pb.bento.uri.uri)
        except BentoMLException as error:
            deployment_pb.state.state = DeploymentState.ERROR
            deployment_pb.state.error_message = f'Error: {str(error)}'
            return ApplyDeploymentResponse(
                status=error.status_proto, deployment=deployment_pb
            )

    def _add(self, deployment_pb, bento_pb, bento_path):
        if loader._is_remote_path(bento_path):
            with loader._resolve_remote_bundle_path(bento_path) as local_path:
                return self._add(deployment_pb, bento_pb, local_path)

        deployment_spec = deployment_pb.spec
        lambda_deployment_config = deployment_spec.aws_lambda_operator_config
        bento_service_metadata = bento_pb.bento.bento_service_metadata
        lambda_s3_bucket = generate_aws_compatible_string(
            'btml-{namespace}-{name}-{random_string}'.format(
                namespace=deployment_pb.namespace,
                name=deployment_pb.name,
                random_string=uuid.uuid4().hex[:6].lower(),
            )
        )
        try:
            create_s3_bucket_if_not_exists(
                lambda_s3_bucket, lambda_deployment_config.region
            )
            _deploy_lambda_function(
                deployment_pb=deployment_pb,
                bento_service_metadata=bento_service_metadata,
                deployment_spec=deployment_spec,
                lambda_s3_bucket=lambda_s3_bucket,
                lambda_deployment_config=lambda_deployment_config,
                bento_path=bento_path,
            )
            return ApplyDeploymentResponse(status=Status.OK(), deployment=deployment_pb)
        except BentoMLException as error:
            if lambda_s3_bucket and lambda_deployment_config:
                cleanup_s3_bucket_if_exist(
                    lambda_s3_bucket, lambda_deployment_config.region
                )
            raise error

    def update(self, deployment_pb, previous_deployment):
        try:
            deployment_spec = deployment_pb.spec
            bento_pb = self.yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            if bento_pb.bento.uri.type not in (BentoUri.LOCAL, BentoUri.S3):
                raise BentoMLException(
                    'BentoML currently not support {} repository'.format(
                        BentoUri.StorageType.Name(bento_pb.bento.uri.type)
                    )
                )

            return self._update(
                deployment_pb, previous_deployment, bento_pb, bento_pb.bento.uri.uri
            )
        except BentoMLException as error:
            deployment_pb.state.state = DeploymentState.ERROR
            deployment_pb.state.error_message = f'Error: {str(error)}'
            return ApplyDeploymentResponse(
                status=error.status_code, deployment_pb=deployment_pb
            )

    def _update(self, deployment_pb, current_deployment, bento_pb, bento_path):
        if loader._is_remote_path(bento_path):
            with loader._resolve_remote_bundle_path(bento_path) as local_path:
                return self._update(
                    deployment_pb, current_deployment, bento_pb, local_path
                )
        updated_deployment_spec = deployment_pb.spec
        updated_lambda_deployment_config = (
            updated_deployment_spec.aws_lambda_operator_config
        )
        updated_bento_service_metadata = bento_pb.bento.bento_service_metadata
        describe_result = self.describe(deployment_pb)
        if describe_result.status.status_code != status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                describe_result.status
            )
            raise YataiDeploymentException(
                f'Failed fetching Lambda deployment current status - '
                f'{error_code}:{error_message}'
            )
        latest_deployment_state = json.loads(describe_result.state.info_json)
        if 's3_bucket' in latest_deployment_state:
            lambda_s3_bucket = latest_deployment_state['s3_bucket']
        else:
            raise BentoMLException(
                'S3 Bucket is missing in the AWS Lambda deployment, please make sure '
                'it exists and try again'
            )

        _deploy_lambda_function(
            deployment_pb=deployment_pb,
            bento_service_metadata=updated_bento_service_metadata,
            deployment_spec=updated_deployment_spec,
            lambda_s3_bucket=lambda_s3_bucket,
            lambda_deployment_config=updated_lambda_deployment_config,
            bento_path=bento_path,
        )

        return ApplyDeploymentResponse(deployment=deployment_pb, status=Status.OK())

    def delete(self, deployment_pb):
        try:
            logger.debug('Deleting AWS Lambda deployment')

            deployment_spec = deployment_pb.spec
            lambda_deployment_config = deployment_spec.aws_lambda_operator_config
            lambda_deployment_config.region = (
                lambda_deployment_config.region or get_default_aws_region()
            )
            if not lambda_deployment_config.region:
                raise InvalidArgument('AWS region is missing')

            cf_client = boto3.client('cloudformation', lambda_deployment_config.region)
            stack_name = generate_aws_compatible_string(
                deployment_pb.namespace, deployment_pb.name
            )
            if deployment_pb.state.info_json:
                deployment_info_json = json.loads(deployment_pb.state.info_json)
                bucket_name = deployment_info_json.get('s3_bucket')
                if bucket_name:
                    cleanup_s3_bucket_if_exist(
                        bucket_name, lambda_deployment_config.region
                    )

            logger.debug(
                'Deleting AWS CloudFormation: %s that includes Lambda function '
                'and related resources',
                stack_name,
            )
            cf_client.delete_stack(StackName=stack_name)
            return DeleteDeploymentResponse(status=Status.OK())

        except BentoMLException as error:
            return DeleteDeploymentResponse(status=error.status_proto)

    def describe(self, deployment_pb):
        try:
            deployment_spec = deployment_pb.spec
            lambda_deployment_config = deployment_spec.aws_lambda_operator_config
            lambda_deployment_config.region = (
                lambda_deployment_config.region or get_default_aws_region()
            )
            if not lambda_deployment_config.region:
                raise InvalidArgument('AWS region is missing')

            bento_pb = self.yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            bento_service_metadata = bento_pb.bento.bento_service_metadata
            api_names = (
                [lambda_deployment_config.api_name]
                if lambda_deployment_config.api_name
                else [api.name for api in bento_service_metadata.apis]
            )

            try:
                cf_client = boto3.client(
                    'cloudformation', lambda_deployment_config.region
                )
                stack_name = generate_aws_compatible_string(
                    '{ns}-{name}'.format(
                        ns=deployment_pb.namespace, name=deployment_pb.name
                    )
                )
                cloud_formation_stack_result = cf_client.describe_stacks(
                    StackName=stack_name
                )
                stack_result = cloud_formation_stack_result.get('Stacks')[0]
                # https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/\
                # using-cfn-describing-stacks.html
                success_status = ['CREATE_COMPLETE', 'UPDATE_COMPLETE']
                if stack_result['StackStatus'] in success_status:
                    if stack_result.get('Outputs'):
                        outputs = stack_result['Outputs']
                    else:
                        return DescribeDeploymentResponse(
                            status=Status.ABORTED('"Outputs" field is not present'),
                            state=DeploymentState(
                                state=DeploymentState.ERROR,
                                error_message='"Outputs" field is not present',
                            ),
                        )
                elif stack_result['StackStatus'] in FAILED_CLOUDFORMATION_STACK_STATUS:
                    state = DeploymentState(state=DeploymentState.FAILED)
                    state.timestamp.GetCurrentTime()
                    return DescribeDeploymentResponse(status=Status.OK(), state=state)
                else:
                    state = DeploymentState(state=DeploymentState.PENDING)
                    state.timestamp.GetCurrentTime()
                    return DescribeDeploymentResponse(status=Status.OK(), state=state)
            except Exception as error:  # pylint: disable=broad-except
                state = DeploymentState(
                    state=DeploymentState.ERROR, error_message=str(error)
                )
                state.timestamp.GetCurrentTime()
                return DescribeDeploymentResponse(
                    status=Status.INTERNAL(str(error)), state=state
                )
            outputs = {o['OutputKey']: o['OutputValue'] for o in outputs}
            info_json = {}

            if 'EndpointUrl' in outputs:
                info_json['endpoints'] = [
                    outputs['EndpointUrl'] + '/' + api_name for api_name in api_names
                ]
            if 'S3Bucket' in outputs:
                info_json['s3_bucket'] = outputs['S3Bucket']

            state = DeploymentState(
                state=DeploymentState.RUNNING, info_json=json.dumps(info_json)
            )
            state.timestamp.GetCurrentTime()
            return DescribeDeploymentResponse(status=Status.OK(), state=state)
        except BentoMLException as error:
            return DescribeDeploymentResponse(status=error.status_proto)
