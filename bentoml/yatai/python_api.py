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
import json
import logging
import tarfile
import requests
import tempfile


from bentoml.service import BentoService
from bentoml.exceptions import BentoMLException
from bentoml.proto.repository_pb2 import (
    AddBentoRequest,
    BentoUri,
    UpdateBentoRequest,
    UploadStatus,
)
from bentoml.proto.status_pb2 import Status
from bentoml.utils.usage_stats import track_save
from bentoml.archive import save_to_dir


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

    if not isinstance(bento_service, BentoService):
        raise BentoMLException(
            "Only instance of custom BentoService class can be saved or uploaded"
        )

    if version is not None:
        bento_service.set_version(version)

    # if base_path is not None, default repository base path in config will be override
    if base_path is not None:
        logger.warning("Overriding default repository path to '%s'", base_path)

    from bentoml.yatai import get_yatai_service

    yatai = get_yatai_service(repo_base_url=base_path)

    request = AddBentoRequest(
        bento_name=bento_service.name, bento_version=bento_service.version
    )
    response = yatai.AddBento(request)

    if response.status.status_code != Status.OK:
        raise BentoMLException(
            "Error adding bento to repository: {}:{}".format(
                response.status.status_code, response.status.error_message
            )
        )

    if response.uri.type == BentoUri.LOCAL:
        # Saving directory to path managed by LocalBentoRepository
        save_to_dir(bento_service, response.uri.uri)

        update_bento_upload_progress(yatai, bento_service)

        # Return URI to saved bento in repository storage
        return response.uri.uri
    elif response.uri.type == BentoUri.S3:
        with tempfile.TemporaryDirectory() as tmpdir:
            update_bento_upload_progress(
                yatai, bento_service, UploadStatus.UPLOADING, 0
            )
            save_to_dir(bento_service, tmpdir)

            fileobj = io.BytesIO()
            with tarfile.open(mode="w:gz", fileobj=fileobj) as tar:
                tar.add(tmpdir, arcname=bento_service.name)
            fileobj.seek(0, 0)

            files = {'file': ('dummy', fileobj)}  # dummy file name because file name
            # has been generated when getting the pre-signed signature.
            http_response = requests.post(
                response.uri.uri,
                data=json.loads(response.uri.additional_fields),
                files=files,
            )

            if http_response.status_code != 204:
                update_bento_upload_progress(yatai, bento_service, UploadStatus.ERROR)

                raise BentoMLException(
                    "Error saving Bento to S3 with status code {} and error detail "
                    "is {}".format(http_response.status_code, http_response.text)
                )

            logger.info(
                "Successfully saved Bento '%s:%s' to S3: %s",
                bento_service.name,
                bento_service.version,
                response.uri.uri,
            )

            update_bento_upload_progress(yatai, bento_service)

            return response.uri.uri

    else:
        raise BentoMLException(
            "Error saving Bento to target repository, URI type %s at %s not supported"
            % response.uri.type,
            response.uri.uri,
        )


def update_bento_upload_progress(
    yatai, bento_service, status=UploadStatus.DONE, progress=None
):
    upload_status = UploadStatus(status=status)
    upload_status.updated_at.GetCurrentTime()
    update_bento_req = UpdateBentoRequest(
        bento_name=bento_service.name,
        bento_version=bento_service.version,
        upload_status=upload_status,
        service_metadata=bento_service._get_bento_service_metadata_pb(),
    )
    yatai.UpdateBento(update_bento_req)
