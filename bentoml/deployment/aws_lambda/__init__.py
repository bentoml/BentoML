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

import json
import os
import logging
import uuid

from botocore.exceptions import ClientError

import boto3
from ruamel.yaml import YAML

from bentoml.bundler import loader
from bentoml.deployment.aws_lambda.utils import (
    ensure_sam_available_or_raise,
    init_sam_project,
    lambda_deploy,
    lambda_package,
    validate_lambda_template,
)
from bentoml.deployment.operator import DeploymentOperatorBase
from bentoml.deployment.utils import (
    exception_to_return_status,
    ensure_docker_available_or_raise,
    generate_aws_compatible_string,
    raise_if_api_names_not_found_in_bento_service_metadata,
)
from bentoml.exceptions import BentoMLException
from bentoml.proto.deployment_pb2 import (
    ApplyDeploymentResponse,
    Deployment,
    DeploymentState,
    DescribeDeploymentResponse,
    DeleteDeploymentResponse,
)
from bentoml.proto.repository_pb2 import GetBentoRequest, BentoUri
from bentoml.utils import Path
from bentoml.utils.s3 import create_s3_bucket_if_not_exists, upload_directory_to_s3
from bentoml.utils.tempdir import TempDirectory
from bentoml.yatai.status import Status


logger = logging.getLogger(__name__)


def _create_aws_lambda_cloudformation_template_file(
    project_dir,
    deployment_name,
    api_names,
    bento_service_name,
    artifacts_prefix,
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
                'Cors': {'AllowOrigin': "'*'"},
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
                'FunctionName': '{deployment}-{api}'.format(
                    deployment=deployment_name, api=api_name
                ),
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
                        'BENTOML_S3_BUCKET': s3_bucket_name,
                        'BENTOML_ARTIFACTS_PREFIX': artifacts_prefix,
                        'BENTOML_API_NAME': api_name,
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
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod"
    Description: URL for endpoint
"""
        )
    return template_file_path


def _cleanup_s3_bucket(bucket_name, region):
    s3_client = boto3.client('s3', region)
    s3 = boto3.resource('s3')
    try:
        logger.debug('Removing all objects inside bucket %s', bucket_name)
        s3.Bucket(bucket_name).objects.all().delete()
        logger.debug('Deleting bucket %s', bucket_name)
        s3_client.delete_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response and e.response['Error']['Code'] == 'NoSuchBucket':
            # If there is no bucket, we just let it silently fail, dont have to do
            # any thing
            return
        else:
            raise e


class AwsLambdaDeploymentOperator(DeploymentOperatorBase):
    def apply(self, deployment_pb, yatai_service, prev_deployment):
        try:
            ensure_sam_available_or_raise()
            ensure_docker_available_or_raise()
            deployment_spec = deployment_pb.spec

            bento_pb = yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            if bento_pb.bento.uri.type not in (BentoUri.LOCAL, BentoUri.S3):
                raise BentoMLException(
                    'BentoML currently not support {} repository'.format(
                        bento_pb.bento.uri.type
                    )
                )

            return self._apply(
                deployment_pb,
                bento_pb,
                yatai_service,
                bento_pb.bento.uri.uri,
                prev_deployment,
            )
        except BentoMLException as error:
            return ApplyDeploymentResponse(status=exception_to_return_status(error))

    def _apply(
        self, deployment_pb, bento_pb, yatai_service, bento_path, prev_deployment
    ):
        if loader._is_remote_path(bento_path):
            with loader._resolve_remote_bundle_path(bento_path) as local_path:
                return self._apply(
                    deployment_pb, bento_pb, yatai_service, local_path, prev_deployment
                )

        try:
            deployment_spec = deployment_pb.spec
            aws_config = deployment_spec.aws_lambda_operator_config
            bento_service_metadata = bento_pb.bento.bento_service_metadata

            py_major, py_minor, _ = bento_service_metadata.env.python_version.split('.')
            if py_major != '3':
                raise BentoMLException(
                    'Python 2 is not supported for Lambda Deployment'
                )
            python_runtime = 'python{}.{}'.format(py_major, py_minor)

            api_names = (
                [aws_config.api_name]
                if aws_config.api_name
                else [api.name for api in bento_service_metadata.apis]
            )

            raise_if_api_names_not_found_in_bento_service_metadata(
                bento_service_metadata, api_names
            )

            if prev_deployment:
                info_json = json.loads(prev_deployment.state.info_json)
                lambda_s3_bucket = info_json['s3_bucket']
            else:
                lambda_s3_bucket = generate_aws_compatible_string(
                    'btml-{namespace}-{name}-{random_string}'.format(
                        namespace=deployment_pb.namespace,
                        name=deployment_pb.name,
                        random_string=uuid.uuid4().hex[:6].lower(),
                    )
                )
                create_s3_bucket_if_not_exists(lambda_s3_bucket, aws_config.region)
            deployment_path_prefix = os.path.join(
                deployment_pb.namespace, deployment_pb.name
            )

            logger.debug('Uploading artifacts to S3 bucket')
            artifacts_prefix = os.path.join(
                deployment_path_prefix,
                'artifacts',
                deployment_spec.bento_name,
                deployment_spec.bento_version,
            )
            if bento_service_metadata.artifacts:
                logger.debug(
                    'This lambda deployment requires uploading artifacts to s3'
                )
                upload_dir_path = os.path.join(
                    bento_path, deployment_spec.bento_name, 'artifacts'
                )
                upload_directory_to_s3(
                    upload_dir_path,
                    aws_config.region,
                    lambda_s3_bucket,
                    artifacts_prefix,
                )
            with TempDirectory() as lambda_project_dir:
                logger.debug(
                    'Generating cloudformation template.yaml for lambda project at %s',
                    lambda_project_dir,
                )
                template_file_path = _create_aws_lambda_cloudformation_template_file(
                    project_dir=lambda_project_dir,
                    deployment_name=deployment_pb.name,
                    api_names=api_names,
                    bento_service_name=deployment_spec.bento_name,
                    artifacts_prefix=artifacts_prefix,
                    s3_bucket_name=lambda_s3_bucket,
                    py_runtime=python_runtime,
                    memory_size=aws_config.memory_size,
                    timeout=aws_config.timeout,
                )
                validate_lambda_template(template_file_path)
                logger.debug('Lambda project directory: {}'.format(lambda_project_dir))
                logger.debug('initializing lambda project')
                init_sam_project(
                    lambda_project_dir,
                    bento_path,
                    deployment_pb.name,
                    deployment_spec.bento_name,
                    api_names,
                )
                logger.info('Packaging lambda project')
                lambda_package(
                    lambda_project_dir, lambda_s3_bucket, deployment_path_prefix
                )
                logger.info('Deploying lambda project')
                try:
                    lambda_deploy(
                        lambda_project_dir,
                        stack_name=deployment_pb.namespace + '-' + deployment_pb.name,
                    )
                except Exception as e:
                    _cleanup_s3_bucket(lambda_s3_bucket, aws_config.region)
                    return ApplyDeploymentResponse(status=Status.INTERNAL(str(e)))

            logger.info('Finish deployed lambda project, fetching latest status')
            res_deployment_pb = Deployment(state=DeploymentState())
            res_deployment_pb.CopyFrom(deployment_pb)
            state = self.describe(res_deployment_pb, yatai_service).state
            res_deployment_pb.state.CopyFrom(state)
            return ApplyDeploymentResponse(
                status=Status.OK(), deployment=res_deployment_pb
            )

        except BentoMLException as error:
            return ApplyDeploymentResponse(status=exception_to_return_status(error))

    def delete(self, deployment_pb, yatai_service):
        try:
            logger.debug('Deleting AWS Lambda deployment')
            state = self.describe(deployment_pb, yatai_service).state
            if state.state != DeploymentState.RUNNING:
                message = (
                    'Failed to delete, no active deployment {name}. '
                    'The current state is {state}'.format(
                        name=deployment_pb.name,
                        state=DeploymentState.State.Name(state.state),
                    )
                )
                return DeleteDeploymentResponse(status=Status.ABORTED(message))

            deployment_spec = deployment_pb.spec
            aws_config = deployment_spec.aws_lambda_operator_config

            cf_client = boto3.client('cloudformation', aws_config.region)
            stack_name = deployment_pb.namespace + '-' + deployment_pb.name
            cloud_formation_stack_result = cf_client.describe_stacks(
                StackName='{ns}-{name}'.format(
                    ns=deployment_pb.namespace, name=deployment_pb.name
                )
            )
            stack_result = cloud_formation_stack_result.get('Stacks')[0]
            outputs = stack_result['Outputs']
            bucket_name = ''
            for output in outputs:
                if output['OutputKey'] == 'S3Bucket':
                    bucket_name = output['OutputValue']
            if bucket_name:
                _cleanup_s3_bucket(bucket_name, aws_config.region)

            logger.debug('Deleting Lambda function and related resources')
            cf_client.delete_stack(StackName=stack_name)
            return DeleteDeploymentResponse(status=Status.OK())

        except BentoMLException as error:
            return DeleteDeploymentResponse(status=exception_to_return_status(error))

    def describe(self, deployment_pb, yatai_service):
        try:
            deployment_spec = deployment_pb.spec
            aws_config = deployment_spec.aws_lambda_operator_config
            info_json = {'endpoints': []}

            bento_pb = yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            bento_service_metadata = bento_pb.bento.bento_service_metadata
            api_names = (
                [aws_config.api_name]
                if aws_config.api_name
                else [api.name for api in bento_service_metadata.apis]
            )

            try:
                cf_client = boto3.client('cloudformation', aws_config.region)
                cloud_formation_stack_result = cf_client.describe_stacks(
                    StackName='{ns}-{name}'.format(
                        ns=deployment_pb.namespace, name=deployment_pb.name
                    )
                )
                stack_result = cloud_formation_stack_result.get('Stacks')[0]
                if (
                    stack_result['StackStatus'] == 'CREATE_COMPLETE'
                    or stack_result['StackStatus'] == 'UPDATE_COMPLETE'
                ):
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
                else:
                    state = DeploymentState(state=DeploymentState.PENDING)
                    state.timestamp.GetCurrentTime()
                    return DescribeDeploymentResponse(status=Status.OK(), state=state)
            except Exception as error:
                state = DeploymentState(
                    state=DeploymentState.ERROR, error_message=str(error)
                )
                state.timestamp.GetCurrentTime()
                return DescribeDeploymentResponse(
                    status=Status.INTERNAL(str(error)), state=state
                )

            base_url = ''
            s3_bucket = ''
            for output in outputs:
                if output['OutputKey'] == 'EndpointUrl':
                    base_url = output['OutputValue']
                if output['OutputKey'] == 'S3Bucket':
                    s3_bucket = output['OutputValue']
            if base_url:
                info_json['endpoints'] = [
                    base_url + '/' + api_name for api_name in api_names
                ]
            if s3_bucket:
                info_json['s3_bucket'] = s3_bucket
            state = DeploymentState(
                state=DeploymentState.RUNNING, info_json=json.dumps(info_json)
            )
            state.timestamp.GetCurrentTime()
            return DescribeDeploymentResponse(status=Status.OK(), state=state)
        except BentoMLException as error:
            return DescribeDeploymentResponse(status=exception_to_return_status(error))
