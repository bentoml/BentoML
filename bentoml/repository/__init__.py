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
import boto3
from six import add_metaclass
from abc import abstractmethod, ABCMeta
from six.moves.urllib.parse import urlparse


from bentoml import config
from bentoml.exceptions import BentoMLRepositoryException
from bentoml.utils.s3 import is_s3_url
from bentoml.utils import Path
from bentoml.proto.repository_pb2 import BentoUri


@add_metaclass(ABCMeta)
class BentoRepositoryBase(object):
    """
    BentoRepository is the interface for managing saved Bentos over file system or
    cloud storage systems.

    A Bento is a BentoService serialized into a standard file format that can be
    easily load back to a Python session, installed as PyPI package, or run in Conda
    or docker environment with all dependencies configured automatically
    """

    @abstractmethod
    def add(self, bento_name, bento_version):
        """
        Proposing to add a saved BentoService to target repository by providing the
        bento name and version.
        Return value is an URL(file path or s3 path for example), that is ready for
        the client to upload saved Bento files.
        """

    @abstractmethod
    def get(self, bento_name, bento_version):
        """
        Get a file path to the saved Bento files, path must be accessible form local
        machine either through NFS or pre-downloaded to local machine
        """

    @abstractmethod
    def dangerously_delete(self, bento_name, bento_version):
        """
        Deleting the Bento files that was added to this repository earlier, this may
        break existing deployments or create issues when doing deployment rollback
        """


class _LocalBentoRepository(BentoRepositoryBase):
    def __init__(self, base_url):
        if not os.path.exists(base_url):
            # make sure local repo base path exist
            os.mkdir(base_url)

        self.base_path = base_url
        self.uri_type = BentoUri.LOCAL

    def add(self, bento_name, bento_version):
        # Full path containing saved BentoArchive, it the base path with service name
        # and service version as prefix. e.g.:
        # with base_path = '/tmp/my_bento_archive/', the saved bento will resolve in
        # the directory: '/tmp/my_bento_archive/service_name/version/'
        target_dir = os.path.join(self.base_path, bento_name, bento_version)

        # Ensure parent directory exist
        Path(os.path.join(self.base_path), bento_name).mkdir(
            parents=True, exist_ok=True
        )

        # Raise if target bento version already exist in storage
        if os.path.exists(target_dir):
            raise BentoMLRepositoryException(
                "Existing Bento {name}:{version} found in archive: {target_dir}".format(
                    name=bento_name, version=bento_version, target_dir=target_dir
                )
            )

        # Create target directory for upload
        os.mkdir(target_dir)

        return BentoUri(type=self.uri_type, uri=target_dir)

    def get(self, bento_name, bento_version):
        saved_path = os.path.join(self.base_path, bento_name, bento_version)
        if not os.path.exists(saved_path):
            raise BentoMLRepositoryException(
                "Bento {}:{} not found in target repository".format(
                    bento_name, bento_version
                )
            )
        return saved_path

    def dangerously_delete(self, bento_name, bento_version):
        saved_path = os.path.join(self.base_path, bento_name, bento_version)
        return shutil.rmtree(saved_path)


class _S3BentoRepository(BentoRepositoryBase):
    def __init__(self, base_url):
        self.uri_type = BentoUri.S3

        parse_result = urlparse(base_url)
        self.bucket = parse_result.netloc
        self.base_path = parse_result.path

        self.s3_client = boto3.client("s3")

    def add(self, bento_name, bento_version):
        # Generate pre-signed s3 path for upload
        expiration = config('yatai').getint('bento_uri_default_expiration')

        object_name = "/".join([self.base_path, bento_name, bento_version])

        try:
            response = self.s3_client.generate_presigned_post(
                self.bucket,
                object_name,
                Fields=None,
                Conditions=None,
                ExpiresIn=expiration,
            )
        except Exception as e:
            raise BentoMLRepositoryException(
                "Not able to get pre-signed URL on S3. Error: {}".format(e)
            )
        return response

    def get(self, bento_name, bento_version):
        # Return s3 path containing uploaded Bento files
        raise NotImplementedError

    def dangerously_delete(self, bento_name, bento_version):
        # Remove s3 path containing related Bento files
        raise NotImplementedError


class BentoRepository(BentoRepositoryBase):
    def __init__(self, base_url=None):
        if base_url is None:
            base_url = config().get('default_repository_base_url')

        if is_s3_url(base_url):
            self._repo = _S3BentoRepository(base_url)
        else:
            self._repo = _LocalBentoRepository(base_url)

    def add(self, bento_name, bento_version):
        return self._repo.add(bento_name, bento_version)

    def get(self, bento_name, bento_version):
        return self._repo.get(bento_name, bento_version)

    def dangerously_delete(self, bento_name, bento_version):
        return self._repo.dangerously_delete(bento_name, bento_version)
