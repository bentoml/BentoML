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

from ruamel.yaml import YAML

from bentoml.deployment.aws_lambda.utils import (
    ensure_sam_available_or_raise,
    upload_artifacts_to_s3,
    init_sam_project,
    lambda_build,
    lambda_deploy,
    lambda_package,
    generate_framework_layers,
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
)
from bentoml.proto.repository_pb2 import GetBentoRequest, BentoUri
from bentoml.utils import Path
from bentoml.utils.tempdir import TempDirectory
from bentoml.yatai.status import Status


def generate_function_resource(api_name, artifact_bucket_name):
    return {
        'Type': 'AWS::Serverless::Function',
        'Properties': {
            'Runtime': 'python3.7',
            'CodeUri': '',
            'Handler': 'app.{}'.format(api_name),
            'FunctionName': api_name,
            'Timeout': 60,
            'MemorySize': 3008,
            'Events': {
                'Api': {
                    'Type': 'Api',
                    'Properties': {'Path': '/{}'.format(api_name), 'Method': 'get'},
                }
            },
            'Policies': [{'S3ReadPolicy': {'BucketName': artifact_bucket_name}}],
        },
    }


def generate_aws_lambda_template_config(project_dir, api_names, s3_bucket_name):
    template_file_path = os.path.join(project_dir, 'template.yaml')
    yaml = YAML()
    sam_config = {
        'AWSTemplateFormatVersion': '2010-09-09',
        'Transform': 'AWS::Serverless-2016-10-31',
        'Globals': {'Function': {'Timeout': 180, 'Runtime': 'python3.7'}},
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
            api_name, s3_bucket_name
        )

    yaml.dump(sam_config, Path(template_file_path))


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

            api_names = (
                [aws_config.api_name]
                if aws_config.api_name
                else [api.name for api in bento_service_metadata.apis]
            )
            ensure_deploy_api_name_exists_in_bento(
                [api.name for api in bento_service_metadata.apis], api_names
            )
            artifact_bucket_name = '{name}-{bento_name}-{bento_version}'.format(
                name=deployment_pb.name,
                bento_name=deployment_spec.bento_name,
                bento_version=deployment_spec.bento_version,
            )
            upload_artifacts_to_s3(
                aws_config.region,
                artifact_bucket_name,
                bento_archive_path,
                deployment_spec.bento_name,
            )
            with TempDirectory() as lambda_project_dir:
                generate_aws_lambda_template_config(
                    lambda_project_dir, api_names, artifact_bucket_name
                )
                init_sam_project(
                    lambda_project_dir,
                    bento_archive_path,
                    deployment_spec.bento_name,
                    api_names,
                )
                lambda_package(lambda_project_dir, 's3_location')
                lambda_deploy(
                    lambda_project_dir,
                    stack_name=deployment_pb.namespace + '-' + deployment_pb.name,
                )

            res_deployment_pb = Deployment(state=DeploymentState())
            res_deployment_pb.CopyFrom(deployment_pb)
            state = self.describe(res_deployment_pb, yatai_service).state
            res_deployment_pb.state.CopyFrom(state)
            return ApplyDeploymentResponse(
                status=Status.OK(), deployment=res_deployment_pb
            )

        except BentoMLException as error:
            return ApplyDeploymentResponse(status=exception_to_return_status(error))

    def describe(self, deployment_pb, yatai_service):
        pass

    def delete(self, deployment_pb, yatai_service):
        pass
