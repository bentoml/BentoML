from bentoml._internal.utils.lazy_loader import LazyLoader

yatai_proto = LazyLoader('yatai_proto', globals(), 'yatai.proto')


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


class YataiException(Exception):
    """
    Base class for Yatai errors
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
    def http_status_code(self):
        """HTTP response status code"""
        return _proto_status_code_to_http_status_code(self.proto_status_code, 500)


class YataiConfigurationException(YataiException):
    """
    Raises when yatai configuration failed.
    """


class YataiRepositoryException(YataiException):
    """
    Raises when repository failed.
    """
