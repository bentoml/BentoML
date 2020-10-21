import logging
import os

from bentoml.exceptions import BentoMLException
from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.utils.lazy_loader import LazyLoader
from bentoml.utils.usage_stats import (
    track,
    track_load_start,
    track_load_finish,
    track_save,
)
from bentoml.yatai.client import YataiClient
from bentoml.yatai.yatai_service import get_yatai_service
from bentoml.saved_bundle.loader import load_from_bundle_path, _is_remote_path
from bentoml.saved_bundle import load_bento_service_metadata

logger = logging.getLogger(__name__)
yatai_proto = LazyLoader('yatai_proto', globals(), 'bentoml.yatai.proto')


def save(bento_service, yatai_url=None, version=None, labels=None):
    """
    Save and register the given BentoService via BentoML's built-in model management
    system. BentoML by default keeps track of all the SavedBundle's files and metadata
    in local file system under the $BENTOML_HOME(~/bentoml) directory. Users can also
    configure BentoML to save their BentoService to a shared Database and cloud object
    storage such as AWS S3.

    :param bento_service: target BentoService instance to be saved
    :param yatai_url: optional - URL path to yatai server
    :param version: optional - save with version override
    :param labels: optional - label dictionary
    :return: saved_path: file path to where the BentoService is saved
    """

    if yatai_url:
        yatai_service = get_yatai_service(channel_address=yatai_url)
        yatai_client = YataiClient(yatai_service)
    else:
        yatai_client = YataiClient()

    track_save(bento_service)
    return yatai_client.repository.upload(bento_service, version, labels)


def load(bento, yatai_url=None):
    """
    Load a BentoService instance base on the BentoService tag (key:version) or path.

    Args:
        bento: a BentoService identifier in the format of NAME:VERSION or a path like string
        yatai_url: optional. a YataiService URL address.
    Returns:
        BentoService instance
    """
    track_load_start()
    if _is_remote_path(bento) or os.path.isdir(bento):
        if yatai_url:
            logger.info('Path to BentoService is provided, ignoring the yatai url.')
        svc = load_from_bundle_path(bento)
        track_load_finish(svc)
        return svc
    else:
        if ':' not in bento:
            raise BentoMLException(
                'BentoService name or version is missing. Please provide in the '
                'format of name:version'
            )
        if yatai_url:
            yatai_service = get_yatai_service(channel_address=yatai_url)
            yatai_client = YataiClient(yatai_service)
        else:
            yatai_client = YataiClient()

        name, version = bento.split(':')
        get_bento_result = yatai_client.repository.get(name, version)
        if get_bento_result.status.status_code != yatai_proto.status_pb2.Status.OK:
            error_code, error_message = status_pb_to_error_code_and_message(
                get_bento_result.status
            )
            raise BentoMLException(
                f'BentoService {name}:{version} not found - '
                f'{error_code}:{error_message}'
            )
        if get_bento_result.bento.uri.s3_presigned_url:
            saved_bundle_path = get_bento_result.bento.uri.s3_presigned_url
        elif get_bento_result.bento.uri.gcs_presigned_url:
            saved_bundle_path = get_bento_result.bento.uri.gcs_presigned_url
        else:
            saved_bundle_path = get_bento_result.bento.uri.uri
        svc = load_from_bundle_path(saved_bundle_path)
        track_load_finish(svc)
        return svc


def delete(bento, yatai_url=None):
    track('delete')
    if ':' not in bento:
        raise BentoMLException(
            'BentoService name or version is missing. Please provide in the '
            'format of name:version'
        )
    if yatai_url:
        yatai_service = get_yatai_service(channel_address=yatai_url)
        yatai_client = YataiClient(yatai_service)
    else:
        yatai_client = YataiClient()
    name, version = bento.split(':')
    result = yatai_client.repository.dangerously_delete_bento(
        name=name, version=version
    )
    if result.status.status_code != yatai_proto.status_pb2.Status.OK:
        error_code, error_message = status_pb_to_error_code_and_message(result.status)
        raise BentoMLException(
            f'Failed to delete Bento {bento} {error_code}:{error_message}'
        )


def prune(yatai_url=None, bento_name=None, labels=None):
    track('prune')
    if yatai_url:
        yatai_service = get_yatai_service(channel_address=yatai_url)
        yatai_client = YataiClient(yatai_service)
    else:
        yatai_client = YataiClient()

    list_bentos_result = yatai_client.repository.list(
        bento_name=bento_name, labels=labels,
    )
    if list_bentos_result.status.status_code != yatai_proto.status_pb2.Status.OK:
        error_code, error_message = status_pb_to_error_code_and_message(
            list_bentos_result.status
        )
        raise BentoMLException(f'{error_code}:{error_message}')
    for bento in list_bentos_result.bentos:
        bento_tag = f'{bento.name}:{bento.version}'
        try:
            delete(bento_tag, yatai_url=yatai_url)
        except BentoMLException as e:
            logger.error(f'Failed to delete Bento {bento_tag}: {e}')


def push(bento, yatai_url, labels=None):
    """
    Push(save/register) a local BentoService to a remote yatai server.
    Args:
        bento: a BentoService identifier in the format of NAME:VERSION
        yatai_url: a YataiService URL address
        labels: optional. List of labels for the BentoService.

    Returns:
        BentoService saved path
    """
    track('push')
    loaded_bento_service = load(bento)

    yatai_service = get_yatai_service(channel_address=yatai_url)
    yatai_client = YataiClient(yatai_service)

    return yatai_client.repository.upload(
        loaded_bento_service, loaded_bento_service.version, labels
    )


def pull(bento, yatai_url):
    """
    Pull a BentoService from a yatai service. The BentoService will be saved and
    registered with local yatai service.

    Args:
        bento: a BentoService identifier in the form of NAME:VERSION
        yatai_url: a YataiService URL address

    Returns:
        BentoService saved path
    """
    track('pull')
    loaded_bento_service = load(bento, yatai_url)
    return save(loaded_bento_service)


def get_bento(bento, yatai_url=None):
    """
    Get a BentoService metadata information.
    Args:
        bento: a BentoService identifier in the form of NAME:VERSION
        yatai_url:  a YataiService URL address

    Returns:
        BentoService Metadata - bentoml.yatai.proto.repository_pb2.BentoServiceMetadata
    """
    track('get-bento-info')
    if ':' not in bento:
        raise BentoMLException(
            'BentoService name or version is missing. Please provide in the '
            'format of name:version'
        )
    if yatai_url:
        yatai_service = get_yatai_service(channel_address=yatai_url)
        yatai_client = YataiClient(yatai_service)
    else:
        yatai_client = YataiClient()

    name, version = bento.split(':')
    get_bento_result = yatai_client.repository.get(name, version)
    if get_bento_result.status.status_code != yatai_proto.status_pb2.Status.OK:
        error_code, error_message = status_pb_to_error_code_and_message(
            get_bento_result.status
        )
        raise BentoMLException(
            f'BentoService {name}:{version} not found - '
            f'{error_code}:{error_message}'
        )
    return get_bento_result.bento
    # if get_bento_result.bento.uri.s3_presigned_url:
    #     # Use s3 presigned URL for downloading the repository if it is presented
    #     saved_bundle_path = get_bento_result.bento.uri.s3_presigned_url
    # elif get_bento_result.bento.uri.gcs_presigned_url:
    #     saved_bundle_path = get_bento_result.bento.uri.gcs_presigned_url
    # else:
    #     saved_bundle_path = get_bento_result.bento.uri.uri
    # return load_bento_service_metadata(saved_bundle_path)


def list_bentos(
    bento_name=None,
    limit=None,
    labels=None,
    offset=None,
    order_by=None,
    ascending_order=False,
    yatai_url=None,
):
    """
    List BentoServices that satisfy the specified criteria.
    Args:
        bento_name: optional. BentoService name
        limit: optional. maximum number of returned results
        labels: optional.
        offset: optional. offset of results
        order_by: optional. order by results
        ascending_order:  optional. direction of results order
        yatai_url: optional. a YataiService URL address

    Returns:
        [bentoml.yatai.proto.repository_pb2.Bento]
    """
    track('list-bentos')
    if yatai_url:
        yatai_service = get_yatai_service(channel_address=yatai_url)
        yatai_client = YataiClient(yatai_service)
    else:
        yatai_client = YataiClient()
    list_bentos_result = yatai_client.repository.list(
        bento_name=bento_name,
        limit=limit,
        offset=offset,
        labels=labels,
        order_by=order_by,
        ascending_order=ascending_order,
    )
    if list_bentos_result.status.status_code != yatai_proto.status_pb2.Status.OK:
        error_code, error_message = status_pb_to_error_code_and_message(
            list_bentos_result.status
        )
        raise BentoMLException(f'{error_code}:{error_message}')
    return list_bentos_result.bentos
