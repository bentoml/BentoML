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


from abc import ABCMeta, abstractmethod


class BaseRepository(object):
    """
    BaseRepository is the interface for managing BentoML saved bundle files over either
     a file system or a cloud blob storage systems such as AWS S3 or MinIO

    A BentoML saved bundle is a standard file format that contains trained model files
    as well as serving endpoint code, input/output spec, dependency specs and deployment
    configs.
    """

    __metaclass__ = ABCMeta

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
