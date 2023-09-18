from __future__ import annotations

from http import HTTPStatus


class BentoMLException(Exception):
    """
    Base class for all BentoML's errors. Each custom exception should be derived from this class.
    """

    error_code = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class StateException(Exception):
    """
    Raised when the state of an object is not valid.
    """

    error_code = HTTPStatus.BAD_REQUEST


class RemoteException(BentoMLException):
    """
    A special exception that is used to wrap the exception from remote server
    """

    def __init__(self, message: str, payload: BentoMLException | None = None):
        self.payload = payload
        super().__init__(message)


class InvalidArgument(BentoMLException):
    """
    Raised when BentoML received unexpected/invalid arguments from CLI arguments, HTTP
    Request, or python API function parameters.
    """

    error_code = HTTPStatus.BAD_REQUEST


class InternalServerError(BentoMLException):
    """
    Raised when BentoML received valid arguments from CLI arguments, HTTP
    Request, or python API function parameters, but got internal issues while
    processing.

    * Note to BentoML developers: raise this exception only when exceptions happend
    in the users' code (runner or service) and want to surface it to the user.
    """


class APIDeprecated(BentoMLException):
    """
    Raised when trying to use deprecated APIs of BentoML
    """


class BadInput(InvalidArgument):
    """Raised when API server receiving bad input request"""

    error_code = HTTPStatus.BAD_REQUEST


class NotFound(BentoMLException):
    """
    Raised when specified resource or name not found
    """

    error_code = HTTPStatus.NOT_FOUND


class UnprocessableEntity(BentoMLException):
    """
    Raised when API server receiving unprocessable entity request
    """

    error_code = HTTPStatus.UNPROCESSABLE_ENTITY


class ServiceUnavailable(BentoMLException):
    """
    Raised when incoming requests exceeds the capacity of a server
    """

    error_code = HTTPStatus.SERVICE_UNAVAILABLE


class BentoMLConfigException(BentoMLException):
    """Raised when BentoML is mis-configured or when required configuration is missing"""


class MissingDependencyException(BentoMLException):
    """
    Raised when BentoML component failed to load required dependencies.

    Some BentoML components have optional dependencies that can be installed as extensions.

    For example, when using the :class:`~bentoml._internal.io_descriptors.json.JSON` IODescriptor,
    ``pydantic`` is considered as an optional feature if users want to use it to validate. BentoML
    will still work without ``pydantic`` installed.
    """


class CLIException(BentoMLException):
    """Raised when CLI encounters an issue"""


class YataiRESTApiClientError(BentoMLException):
    """Raised when communicating with BentoCloud server."""

    pass


class ImportServiceError(BentoMLException):
    """Raised when BentoML failed to import the user's service file."""

    pass
