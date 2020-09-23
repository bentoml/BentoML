from bentoml.exceptions import BentoMLException
from bentoml.utils import status_pb_to_error_code_and_message
from bentoml.utils.lazy_loader import LazyLoader
from bentoml.utils.usage_stats import track
from bentoml.yatai.client import YataiClient
from bentoml.yatai.yatai_service import get_yatai_service
from bentoml.saved_bundle.loader import load as load_bundle
from bentoml.saved_bundle import load_bento_service_metadata

yatai_proto = LazyLoader('yatai_proto', globals(), 'bentoml.yatai.proto')


def save(bento_service, version=None, labels=None, yatai_url=None):
    """
    Save and register the given BentoService. By default, BentoML will be saved in the
    local yatai service. If yatai_url is provided, the BentoService will be saved in the
    remote yatai service instead.

    :param bento_service: target BentoService instance to be saved
    :param version: optional - save with version override
    :param labels: optional - labels for the BentoService
    :return: saved_path: file path to where the BentoService is saved
    """
    if yatai_url:
        yatai_service = get_yatai_service(channel_address=yatai_url)
        yatai_client = YataiClient(yatai_service)
    else:
        yatai_client = YataiClient()

    return yatai_client.repository.upload(bento_service, version, labels)


def load(bento, yatai_url=None):
    """
    Load a BentoService instance base on the BentoService tag (key:version). By default,
    it will load from the local yatai service.

    Args:
        bento: a BentoService identifier in the format of NAME:VERSION
        yatai_url: optional. a YataiService URL address.
    Returns:
        BentoService instance
    """
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
        # Use s3 presigned URL for downloading the repository if it is presented
        saved_bundle_path = get_bento_result.bento.uri.s3_presigned_url
    if get_bento_result.bento.uri.gcs_presigned_url:
        saved_bundle_path = get_bento_result.bento.uri.gcs_presigned_url
    else:
        saved_bundle_path = get_bento_result.bento.uri.uri
    return load_bundle(saved_bundle_path)


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


def get(bento, yatai_url=None):
    """
    Get a BentoService metadata information.
    Args:
        bento: a BentoService identifier in the form of NAME:VERSION
        yatai_url:  a YataiService URL address

    Returns:
        BentoService Metadata - bentoml.yatai.proto.repository_pb2.BentoServiceMetadata
    """
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
        # Use s3 presigned URL for downloading the repository if it is presented
        saved_bundle_path = get_bento_result.bento.uri.s3_presigned_url
    if get_bento_result.bento.uri.gcs_presigned_url:
        saved_bundle_path = get_bento_result.bento.uri.gcs_presigned_url
    else:
        saved_bundle_path = get_bento_result.bento.uri.uri
    return load_bento_service_metadata(saved_bundle_path)


def list(
    bento_name=None,
    limit=None,
    labels=None,
    offset=None,
    order_by=None,
    ascending_order=False,
    yatai_url=None,
):
    """
    List BentoServices that satisfy the specified criterias.
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
