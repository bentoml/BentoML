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


from bentoml.proto.status_pb2 import Status as StatusProto


class BentoMLException(Exception):
    """
    Base class for all BentoML's errors.
    Each custom exception should be derived from this class
    """

    status_code = StatusProto.INTERNAL


class ArtifactLoadingException(BentoMLException):
    pass


class ConfigException(BentoMLException):
    pass


class MissingDependencyException(BentoMLException):
    pass


class InvalidArgumentException(BentoMLException):
    status_code = StatusProto.INVALID_ARGUMENT


class YataiServiceException(BentoMLException):
    pass


class DeploymentException(YataiServiceException):
    pass


class RepositoryException(YataiServiceException):
    pass
