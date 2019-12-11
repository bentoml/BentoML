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

from werkzeug.exceptions import BadRequest

from bentoml.proto import status_pb2


_PROTO_STATUS_CODE_TO_HTTP_STATUS_CODE = {
    status_pb2.Status.INTERNAL: 500,  # Internal Server Error
    status_pb2.Status.INVALID_ARGUMENT: 400,  # "Bad Request"
    status_pb2.Status.NOT_FOUND: 404,  # Not Found
    status_pb2.Status.DEADLINE_EXCEEDED: 408,  # Request Time out
    status_pb2.Status.PERMISSION_DENIED: 401,  # Unauthorized
    status_pb2.Status.UNAUTHENTICATED: 401,  # Unauthorized
}


class BentoMLException(Exception):
    """
    Base class for all BentoML's errors.
    Each custom exception should be derived from this class
    """

    status_code = status_pb2.Status.INTERNAL

    @property
    def status_proto(self):
        return status_pb2.Status(status_code=self.status_code, error_message=str(self))

    @property
    def http_status_code(self):
        return _PROTO_STATUS_CODE_TO_HTTP_STATUS_CODE.get(self.status_code, 500)


class Unauthenticated(BentoMLException):
    status_code = status_pb2.Status.UNAUTHENTICATED


class InvalidArgument(BentoMLException):
    status_code = status_pb2.Status.INVALID_ARGUMENT


class ArtifactLoadingException(BentoMLException):
    pass


class BentoMLConfigException(BentoMLException):
    pass


class MissingDependencyException(BentoMLException):
    pass


class BadInput(InvalidArgument, BadRequest):
    """Raise when BentoHandler receiving bad input request"""

    pass


class YataiServiceException(BentoMLException):
    pass


class YataiServiceRpcAborted(YataiServiceException):
    status_code = status_pb2.Status.ABORTED


class YataiDeploymentException(YataiServiceException):
    pass


class YataiRepositoryException(YataiServiceException):
    pass
