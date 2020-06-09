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


class BentoRepositoryBase(object):
    """
    BentoRepository is the interface for managing saved Bentos over file system or
    cloud storage systems.

    A Bento is a BentoService serialized into a standard file format that can be
    easily load back to a Python session, installed as PyPI package, or run in Conda
    or docker environment with all dependencies configured automatically
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
