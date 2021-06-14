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
import grpc
from bentoml.utils.lazy_loader import LazyLoader

yatai_proto = LazyLoader('yatai_proto', globals(), 'bentoml.yatai.proto')


def _proto_status_code_to_http_status_code(proto_status_code, fallback):
    _PROTO_STATUS_CODE_TO_HTTP_STATUS_CODE = {
        yatai_proto.status_pb2.Status.INTERNAL: 500,  # Internal Server Error
        yatai_proto.status_pb2.Status.INVALID_ARGUMENT: 400,  # "Bad Request"
        yatai_proto.status_pb2.Status.NOT_FOUND: 404,  # Not Found
        yatai_proto.status_pb2.Status.DEADLINE_EXCEEDED: 408,  # Request Time out
        yatai_proto.status_pb2.Status.PERMISSION_DENIED: 401,  # Unauthorized
        yatai_proto.status_pb2.Status.UNAUTHENTICATED: 401,  # Unauthorized
        yatai_proto.status_pb2.Status.FAILED_PRECONDITION: 500,  # Internal Server Error
    }
    return _PROTO_STATUS_CODE_TO_HTTP_STATUS_CODE.get(proto_status_code, fallback)


class BentoMLException(Exception):
    """
    Base class for all BentoML's errors.
    Each custom exception should be derived from this class
    """

    @property
    def proto_status_code(self):
        return yatai_proto.status_pb2.Status.INTERNAL

    @property
    def status_proto(self):
        return yatai_proto.status_pb2.Status(
            status_code=self.proto_status_code, error_message=str(self)
        )

    @property
    def status_code(self):
        """HTTP response status code"""
        return _proto_status_code_to_http_status_code(self.proto_status_code, 500)


class RemoteException(BentoMLException):
    """
    Raise when known exceptions happened in remote server(a model server normally)
    """

    def __init__(self, *args, payload, **kwargs):
        super(RemoteException, self).__init__(*args, **kwargs)
        self.payload = payload


class BentoMLRpcError(BentoMLException):
    def __init__(self, grpc_error, message):
        super(BentoMLRpcError, self).__init__()
        self.grpc_error = grpc_error
        self.message = message
        if self.grpc_error.code == grpc.StatusCode.DEADLINE_EXCEEDED:
            self.grpc_error_message = 'Request time out'
        else:
            self.grpc_error_message = self.grpc_error.details()

    def __str__(self):
        return f'{self.message}: {self.grpc_error_message}'


class Unauthenticated(BentoMLException):
    """
    Raise when a BentoML operation is not authenticated properly, either against 3rd
    party cloud service such as AWS s3, Docker Hub, or Atalaya hosted BentoML service
    """

    @property
    def proto_status_code(self):
        return yatai_proto.status_pb2.Status.UNAUTHENTICATED


class InvalidArgument(BentoMLException):
    """
    Raise when BentoML received unexpected/invalid arguments from CLI arguments, HTTP
    Request, or python API function parameters
    """

    @property
    def proto_status_code(self):
        return yatai_proto.status_pb2.Status.INVALID_ARGUMENT


class APIDeprecated(BentoMLException):
    """
    Raise when trying to use deprecated APIs of BentoML
    """

    @property
    def proto_status_code(self):
        return yatai_proto.status_pb2.Status.INVALID_ARGUMENT


class BadInput(InvalidArgument):
    """Raise when InputAdapter receiving bad input request"""


class NotFound(BentoMLException):
    """
    Raise when specified resource or name not found
    """

    @property
    def proto_status_code(self):
        return yatai_proto.status_pb2.Status.NOT_FOUND


class FailedPrecondition(BentoMLException):
    """
    Raise when required precondition check failed
    """

    @property
    def proto_status_code(self):
        return yatai_proto.status_pb2.Status.FAILED_PRECONDITION


class LockUnavailable(BentoMLException):
    """
    Raise when a bundle/deployment resource is unable to be locked
    """

    @property
    def proto_status_code(self):
        return yatai_proto.status_pb2.Status.FAILED_PRECONDITION


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

    @property
    def proto_status_code(self):
        return yatai_proto.status_pb2.Status.ABORTED


class YataiDeploymentException(YataiServiceException):
    """Raise when YataiService encounters an issue creating/managing deployments"""


class YataiRepositoryException(YataiServiceException):
    """Raise when YataiService encounters an issue managing BentoService repository"""


class AWSServiceError(YataiDeploymentException):
    """Raise when YataiService encounters an issue with AWS service"""


class AzureServiceError(YataiDeploymentException):
    """Raise when YataiService encounters an issue with Azure service"""


class CLIException(BentoMLException):
    """Raise when CLI encounters an issue"""


class YataiLabelException(YataiServiceException):
    """Raise when YataiService encounters an issue managing BentoService label"""
