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

from bentoml.proto import status_pb2


_PROTO_STATUS_CODE_TO_HTTP_STATUS_CODE = {
    status_pb2.Status.INTERNAL: 500,  # Internal Server Error
    status_pb2.Status.INVALID_ARGUMENT: 400,  # "Bad Request"
    status_pb2.Status.NOT_FOUND: 404,  # Not Found
    status_pb2.Status.DEADLINE_EXCEEDED: 408,  # Request Time out
    status_pb2.Status.PERMISSION_DENIED: 401,  # Unauthorized
    status_pb2.Status.UNAUTHENTICATED: 401,  # Unauthorized
    status_pb2.Status.FAILED_PRECONDITION: 500,  # Internal Server Error
}


class BentoMLException(Exception):
    """
    Base class for all BentoML's errors.
    Each custom exception should be derived from this class
    """

    proto_status_code = status_pb2.Status.INTERNAL

    @property
    def status_proto(self):
        return status_pb2.Status(
            status_code=self.proto_status_code, error_message=str(self)
        )

    @property
    def status_code(self):
        """HTTP response status code"""
        return _PROTO_STATUS_CODE_TO_HTTP_STATUS_CODE.get(self.proto_status_code, 500)


class Unauthenticated(BentoMLException):
    """
    Raise when a BentoML operation is not authenticated properly, either against 3rd
    party cloud service such as AWS s3, Docker Hub, or Atalaya hosted BentoML service
    """

    proto_status_code = status_pb2.Status.UNAUTHENTICATED


class InvalidArgument(BentoMLException):
    """
    Raise when BentoML received unexpected/invalid arguments from CLI arguments, HTTP
    Request, or python API function parameters
    """

    proto_status_code = status_pb2.Status.INVALID_ARGUMENT


class BadInput(InvalidArgument):
    """Raise when BentoHandler receiving bad input request"""


class NotFound(BentoMLException):
    """
    Raise when specified resource or name not found
    """

    proto_status_code = status_pb2.Status.NOT_FOUND


class FailedPrecondition(BentoMLException):
    """
    Raise when required precondition check failed
    """

    proto_status_code = status_pb2.Status.FAILED_PRECONDITION


class ArtifactLoadingException(BentoMLException):
    """Raise when BentoService failed to load model artifacts from saved bundle"""


class BentoMLConfigException(BentoMLException):
    """Raise when BentoML is misconfigured or when required configuration is missing"""


class MissingDependencyException(BentoMLException):
    """
    Raise when BentoML component failed to load required dependency - some BentoML
    components has dependency that is optional to the library itself. For example,
    when using SklearnModelArtifact, the scikit-learn module is required although
    BentoML does not require scikit-learn to be a dependency when installed
    """


class YataiServiceException(BentoMLException):
    """Raise when YataiService encounters an error"""


class YataiServiceRpcAborted(YataiServiceException):
    """Raise when YataiService RPC operation aborted"""

    proto_status_code = status_pb2.Status.ABORTED


class YataiDeploymentException(YataiServiceException):
    """Raise when YataiService encounters an issue creating/managing deployments"""


class YataiRepositoryException(YataiServiceException):
    """Raise when YataiService encounters an issue managing BentoService repoistory"""


class AWSServiceError(YataiDeploymentException):
    """Raise when YataiService encounters an issue with AWS service"""
