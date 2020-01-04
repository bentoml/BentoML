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

# List of APIs for accessing remote or local yatai service via Python


import io
import os
import json
import logging
import tarfile
import requests
import shutil

from bentoml import config
from bentoml.deployment.store import ALL_NAMESPACE_TAG
from bentoml.proto.deployment_pb2 import (
    ApplyDeploymentRequest,
    DescribeDeploymentRequest,
    GetDeploymentRequest,
    DeploymentSpec,
    DeleteDeploymentRequest,
    ListDeploymentsRequest,
    ApplyDeploymentResponse,
)
from bentoml.exceptions import BentoMLException, YataiDeploymentException
from bentoml.proto.repository_pb2 import (
    AddBentoRequest,
    GetBentoRequest,
    BentoUri,
    UpdateBentoRequest,
    UploadStatus,
)
from bentoml.proto import status_pb2
from bentoml.utils.usage_stats import track_save
from bentoml.utils.tempdir import TempDirectory
from bentoml.bundler import save_to_dir, load_bento_service_metadata
from bentoml.utils.validator import validate_deployment_pb_schema
from bentoml.yatai.deployment_utils import (
    deployment_yaml_string_to_pb,
    deployment_dict_to_pb,
)
from bentoml.yatai.status import Status

logger = logging.getLogger(__name__)


def upload_bento_service(bento_service, base_path=None, version=None):
    """Save given bento_service via BentoML's default Yatai service, which manages
    all saved Bento files and their deployments in cloud platforms. If remote yatai
    service has not been configured, this will default to saving new Bento to local
    file system under BentoML home directory

    Args:
        bento_service (bentoml.service.BentoService): a Bento Service instance
        base_path (str): optional, base path of the bento repository
        version (str): optional,
    Return:
        URI to where the BentoService is being saved to
    """
    track_save(bento_service)

    with TempDirectory() as tmpdir:
        save_to_dir(bento_service, tmpdir, version)
        return _upload_bento_service(tmpdir, base_path)


def _upload_bento_service(saved_bento_path, base_path):
    bento_service_metadata = load_bento_service_metadata(saved_bento_path)

    from bentoml.yatai import get_yatai_service

    # if base_path is not None, default repository base path in config will be override
    if base_path is not None:
        logger.warning("Overriding default repository path to '%s'", base_path)
    yatai = get_yatai_service(repo_base_url=base_path)

    get_bento_response = yatai.GetBento(
        GetBentoRequest(
            bento_name=bento_service_metadata.name,
            bento_version=bento_service_metadata.version,
        )
    )
    if get_bento_response.status.status_code == status_pb2.Status.OK:
        raise BentoMLException(
            "BentoService bundle {}:{} already registered in repository. Reset "
            "BentoService version with BentoService#set_version or bypass BentoML's "
            "model registry feature with BentoService#save_to_dir".format(
                bento_service_metadata.name, bento_service_metadata.version
            )
        )
    elif get_bento_response.status.status_code != status_pb2.Status.NOT_FOUND:
        raise BentoMLException(
            'Failed accessing YataiService. {error_code}:'
            '{error_message}'.format(
                error_code=Status.Name(get_bento_response.status.status_code),
                error_message=get_bento_response.status.error_message,
            )
        )
    request = AddBentoRequest(
        bento_name=bento_service_metadata.name,
        bento_version=bento_service_metadata.version,
    )
    response = yatai.AddBento(request)

    if response.status.status_code != status_pb2.Status.OK:
        raise BentoMLException(
            "Error adding BentoService bundle to repository: {}:{}".format(
                Status.Name(response.status.status_code), response.status.error_message
            )
        )

    if response.uri.type == BentoUri.LOCAL:
        if os.path.exists(response.uri.uri):
            # due to copytree dst must not already exist
            shutil.rmtree(response.uri.uri)
        shutil.copytree(saved_bento_path, response.uri.uri)

        _update_bento_upload_progress(yatai, bento_service_metadata)

        logger.info(
            "BentoService bundle '%s:%s' created at: %s",
            bento_service_metadata.name,
            bento_service_metadata.version,
            response.uri.uri,
        )
        # Return URI to saved bento in repository storage
        return response.uri.uri
    elif response.uri.type == BentoUri.S3:
        _update_bento_upload_progress(
            yatai, bento_service_metadata, UploadStatus.UPLOADING, 0
        )

        fileobj = io.BytesIO()
        with tarfile.open(mode="w:gz", fileobj=fileobj) as tar:
            tar.add(saved_bento_path, arcname=bento_service_metadata.name)
        fileobj.seek(0, 0)

        files = {'file': ('dummy', fileobj)}  # dummy file name because file name
        # has been generated when getting the pre-signed signature.
        data = json.loads(response.uri.additional_fields)
        uri = data.pop('url')
        http_response = requests.post(uri, data=data, files=files)

        if http_response.status_code != 204:
            _update_bento_upload_progress(
                yatai, bento_service_metadata, UploadStatus.ERROR
            )

            raise BentoMLException(
                "Error saving BentoService bundle to S3. {}: {} ".format(
                    Status.Name(http_response.status_code), http_response.text
                )
            )

        _update_bento_upload_progress(yatai, bento_service_metadata)

        logger.info(
            "Successfully saved BentoService bundle '%s:%s' to S3: %s",
            bento_service_metadata.name,
            bento_service_metadata.version,
            response.uri.uri,
        )

        return response.uri.uri

    else:
        raise BentoMLException(
            "Error saving Bento to target repository, URI type %s at %s not supported"
            % response.uri.type,
            response.uri.uri,
        )


def _update_bento_upload_progress(
    yatai, bento_service_metadata, status=UploadStatus.DONE, percentage=None
):
    upload_status = UploadStatus(status=status, percentage=percentage)
    upload_status.updated_at.GetCurrentTime()
    update_bento_req = UpdateBentoRequest(
        bento_name=bento_service_metadata.name,
        bento_version=bento_service_metadata.version,
        upload_status=upload_status,
        service_metadata=bento_service_metadata,
    )
    yatai.UpdateBento(update_bento_req)


def create_deployment(
    deployment_name,
    namespace,
    bento_name,
    bento_version,
    platform,
    operator_spec,
    labels=None,
    annotations=None,
    yatai_service=None,
):
    if yatai_service is None:
        from bentoml.yatai import get_yatai_service

        yatai_service = get_yatai_service()

    # Make sure there is no active deployment with the same deployment name
    get_deployment_pb = yatai_service.GetDeployment(
        GetDeploymentRequest(deployment_name=deployment_name, namespace=namespace)
    )
    if get_deployment_pb.status.status_code == status_pb2.Status.OK:
        raise YataiDeploymentException(
            'Deployment "{name}" already existed, use Update or Apply for updating '
            'existing deployment, delete the deployment, or use a different deployment '
            'name'.format(name=deployment_name)
        )
    if get_deployment_pb.status.status_code != status_pb2.Status.NOT_FOUND:
        raise YataiDeploymentException(
            'Failed accesing YataiService deployment store. {error_code}:'
            '{error_message}'.format(
                error_code=Status.Name(get_deployment_pb.status.status_code),
                error_message=get_deployment_pb.status.error_message,
            )
        )

    deployment_dict = {
        "name": deployment_name,
        "namespace": namespace or config().get('deployment', 'default_namespace'),
        "labels": labels,
        "annotations": annotations,
        "spec": {
            "bento_name": bento_name,
            "bento_version": bento_version,
            "operator": platform,
        },
    }

    operator = platform.replace('-', '_').upper()
    try:
        operator_value = DeploymentSpec.DeploymentOperator.Value(operator)
    except ValueError:
        return ApplyDeploymentResponse(
            status=Status.INVALID_ARGUMENT('Invalid platform "{}"'.format(platform))
        )
    if operator_value == DeploymentSpec.AWS_SAGEMAKER:
        deployment_dict['spec']['sagemaker_operator_config'] = {
            'region': operator_spec.get('region')
            or config().get('aws', 'default_region'),
            'instance_count': operator_spec.get('instance_count'),
            'instance_type': operator_spec.get('instance_type'),
            'api_name': operator_spec.get('api_name', ''),
        }
        if operator_spec.get('num_of_gunicorn_workers_per_instance'):
            deployment_dict['spec']['sagemaker_operator_config'][
                'num_of_gunicorn_workers_per_instance'
            ] = operator_spec.get('num_of_gunicorn_workers_per_instance')
    elif operator_value == DeploymentSpec.AWS_LAMBDA:
        deployment_dict['spec']['aws_lambda_operator_config'] = {
            'region': operator_spec.get('region')
            or config().get('aws', 'default_region')
        }
        for field in ['api_name', 'memory_size', 'timeout']:
            if operator_spec.get(field):
                deployment_dict['spec']['aws_lambda_operator_config'][
                    field
                ] = operator_spec[field]
    elif operator_value == DeploymentSpec.KUBERNETES:
        deployment_dict['spec']['kubernetes_operator_config'] = {
            'kube_namespace': operator_spec.get('kube_namespace', ''),
            'replicas': operator_spec.get('replicas', 0),
            'service_name': operator_spec.get('service_name', ''),
            'service_type': operator_spec.get('service_type', ''),
        }
    else:
        raise YataiDeploymentException(
            'Platform "{}" is not supported in the current version of '
            'BentoML'.format(platform)
        )

    apply_response = apply_deployment(deployment_dict, yatai_service)

    if apply_response.status.status_code == status_pb2.Status.OK:
        describe_response = describe_deployment(
            deployment_name, namespace, yatai_service
        )
        if describe_response.status.status_code == status_pb2.Status.OK:
            deployment_state = describe_response.state
            apply_response.deployment.state.CopyFrom(deployment_state)
            return apply_response

    return apply_response


# TODO update_deployment is not finished.  It will be working on along with cli command
def update_deployment(deployment_name, namespace, yatai_service=None):
    raise NotImplementedError


def apply_deployment(deployment_info, yatai_service=None):
    if yatai_service is None:
        from bentoml.yatai import get_yatai_service

        yatai_service = get_yatai_service()

    try:
        if isinstance(deployment_info, dict):
            deployment_pb = deployment_dict_to_pb(deployment_info)
        elif isinstance(deployment_info, str):
            deployment_pb = deployment_yaml_string_to_pb(deployment_info)
        else:
            raise YataiDeploymentException(
                'Unexpected argument type, expect deployment info to be str in yaml '
                'format or a dict, instead got: {}'.format(str(type(deployment_info)))
            )

        validation_errors = validate_deployment_pb_schema(deployment_pb)
        if validation_errors:
            return ApplyDeploymentResponse(
                status=Status.INVALID_ARGUMENT(
                    'Failed to validate deployment: {errors}'.format(
                        errors=validation_errors
                    )
                )
            )

        return yatai_service.ApplyDeployment(
            ApplyDeploymentRequest(deployment=deployment_pb)
        )
    except BentoMLException as error:
        return ApplyDeploymentResponse(status=Status.INTERNAL(str(error)))


def describe_deployment(namespace, name, yatai_service=None):
    if yatai_service is None:
        from bentoml.yatai import get_yatai_service

        yatai_service = get_yatai_service()
    return yatai_service.DescribeDeployment(
        DescribeDeploymentRequest(deployment_name=name, namespace=namespace)
    )


def get_deployment(namespace, name, yatai_service=None):
    if yatai_service is None:
        from bentoml.yatai import get_yatai_service

        yatai_service = get_yatai_service()
    return yatai_service.GetDeployment(
        GetDeploymentRequest(deployment_name=name, namespace=namespace)
    )


def delete_deployment(
    deployment_name, namespace, force_delete=False, yatai_service=None
):
    if yatai_service is None:
        from bentoml.yatai import get_yatai_service

        yatai_service = get_yatai_service()

    return yatai_service.DeleteDeployment(
        DeleteDeploymentRequest(
            deployment_name=deployment_name,
            namespace=namespace,
            force_delete=force_delete,
        )
    )


def list_deployments(
    limit=None,
    filters=None,
    labels=None,
    namespace=None,
    is_all_namespaces=False,
    yatai_service=None,
):
    if yatai_service is None:
        from bentoml.yatai import get_yatai_service

        yatai_service = get_yatai_service()

    if is_all_namespaces:
        if namespace is not None:
            logger.warning(
                'Ignoring `namespace=%s` due to the --all-namespace flag presented',
                namespace,
            )
        namespace = ALL_NAMESPACE_TAG

    return yatai_service.ListDeployments(
        ListDeploymentsRequest(
            limit=limit, filter=filters, labels=labels, namespace=namespace
        )
    )
