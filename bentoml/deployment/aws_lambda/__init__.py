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
import subprocess

import boto3
from packaging import version
from ruamel.yaml import YAML

from bentoml.deployment.aws_lambda.utils import (
    ensure_sam_available_or_raise,
    upload_artifacts_to_s3,
    init_sam_project,
    lambda_deploy,
    lambda_package,
)
from bentoml.deployment.operator import DeploymentOperatorBase
from bentoml.deployment.utils import (
    exception_to_return_status,
    ensure_deploy_api_name_exists_in_bento,
    ensure_docker_available_or_raise,
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
from bentoml.utils.tempdir import TempDirectory
from bentoml.yatai.status import Status


def generate_function_resource(
    deployment_name, api_name, artifact_bucket_name, memory_size, timeout
):
    return {
        'Type': 'AWS::Serverless::Function',
        'Properties': {
            'Runtime': 'python3.7',
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
                    'Properties': {'Path': '/{}'.format(api_name), 'Method': 'get'},
                }
            },
            'Policies': [{'S3ReadPolicy': {'BucketName': artifact_bucket_name}}],
        },
    }


def generate_aws_lambda_template_config(
    project_dir,
    deployment_name,
    api_names,
    s3_bucket_name,
    python_runtime,
    memory_size,
    timeout,
):
    template_file_path = os.path.join(project_dir, 'template.yaml')
    yaml = YAML()
    sam_config = {
        'AWSTemplateFormatVersion': '2010-09-09',
        'Transform': 'AWS::Serverless-2016-10-31',
        'Globals': {'Function': {'Timeout': timeout, 'Runtime': python_runtime}},
        'Resources': {},
        # Output section from cloud formation
        'Outputs': {
            'EndpointUrl': {
                'Value': '!Sub "https://${ServerlessRestApi}.'
                'execute-api.${AWS::Region}}.amazonaws.com/Prod"',
                'Description': 'Url for endpoint',
            }
        },
    }
    for api_name in api_names:
        sam_config['Resources'][api_name] = generate_function_resource(
            deployment_name,
            api_name,
            s3_bucket_name,
            memory_size=memory_size,
            timeout=timeout,
        )

    yaml.dump(sam_config, Path(template_file_path))
    try:
        subprocess.check_output(['sam', 'validate'], cwd=project_dir)
    except Exception as error:
        raise BentoMLException(str(error))


class AwsLambdaDeploymentOperator(DeploymentOperatorBase):
    def apply(self, deployment_pb, yatai_service, prev_deployment):
        try:
            ensure_sam_available_or_raise()
            ensure_docker_available_or_raise()
            deployment_spec = deployment_pb.spec
            aws_config = deployment_spec.aws_lambda_operator_config

            bento_pb = yatai_service.GetBento(
                GetBentoRequest(
                    bento_name=deployment_spec.bento_name,
                    bento_version=deployment_spec.bento_version,
                )
            )
            if bento_pb.bento.uri.type != BentoUri.LOCAL:
                raise BentoMLException(
                    'BentoML currently only support local repository'
                )
            else:
                bento_archive_path = bento_pb.bento.uri.uri
            bento_service_metadata = bento_pb.bento.bento_service_metadata

            python_runtime = 'python3.7'
            if version.parse(bento_service_metadata.env.python_version) < version.parse(
                '3.0.0'
            ):
                python_runtime = 'python2.7'

            api_names = (
                [aws_config.api_name]
                if aws_config.api_name
                else [api.name for api in bento_service_metadata.apis]
            )
            ensure_deploy_api_name_exists_in_bento(
                [api.name for api in bento_service_metadata.apis], api_names
            )
            lambda_s3_bucket = '{name}-{bento_name}-{bento_version}'.format(
                name=deployment_pb.name,
                bento_name=deployment_spec.bento_name,
                bento_version=deployment_spec.bento_version,
            )
            upload_artifacts_to_s3(
                aws_config.region,
                lambda_s3_bucket,
                bento_archive_path,
                deployment_spec.bento_name,
            )
            with TempDirectory(cleanup=False) as lambda_project_dir:
                generate_aws_lambda_template_config(
                    lambda_project_dir,
                    deployment_pb.name,
                    api_names,
                    lambda_s3_bucket,
                    python_runtime=python_runtime,
                    memory_size=aws_config.memory_size,
                    timeout=aws_config.timeout,
                )
                init_sam_project(
                    lambda_project_dir,
                    bento_archive_path,
                    deployment_pb.name,
                    deployment_spec.bento_name,
                    api_names,
                )
                print(lambda_project_dir)
                # lambda_package(lambda_project_dir, lambda_s3_bucket)
                # lambda_deploy(
                #     lambda_project_dir,
                #     stack_name=deployment_pb.namespace + '-' + deployment_pb.name,
                # )

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

            s3_client = boto3.client('s3', aws_config.region)
            cf_client = boto3.client('cloudformation', aws_config.region)
            stack_name = deployment_pb.namespace + '-' + deployment_pb.name
            lambda_s3_bucket = '{name}-{bento_name}-{bento_version}'.format(
                name=deployment_pb.name,
                bento_name=deployment_spec.bento_name,
                bento_version=deployment_spec.bento_version,
            )
            delete_s3_result = s3_client.delete_bucket(Bucket=lambda_s3_bucket)
            delete_cf_result = cf_client.delete_stack(StackName=stack_name)
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
                    StackName='{name}-{ns}'.format(
                        ns=deployment_pb.namespace, name=deployment_pb.name
                    )
                )
                outputs = cloud_formation_stack_result.get('Stacks')[0]['Outputs']
            except Exception as error:
                state = DeploymentState(
                    state=DeploymentState.ERROR, error_message=str(error)
                )
                state.timestamp.GetCurrentTime()
                return DescribeDeploymentResponse(
                    status=Status.INTERNAL(str(error)), state=state
                )

            base_url = ''
            for output in outputs:
                if output['OutputKey'] == 'EndpointUrl':
                    base_url = output['OutputValue']
                    break
            if base_url:
                info_json['endpoints'] = [
                    base_url + '/' + api_name for api_name in api_names
                ]
            state = DeploymentState(
                state=DeploymentState.RUNNING, info_json=json.dumps(info_json)
            )
            state.timestamp.GetCurrentTime()
            return DescribeDeploymentResponse(status=Status.OK(), state=state)
        except BentoMLException as error:
            return DescribeDeploymentResponse(status=exception_to_return_status(error))
