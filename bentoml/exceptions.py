from http import HTTPStatus


class BentoMLException(Exception):
    """
    Base class for all BentoML's errors.
    Each custom exception should be derived from this class
    """

    error_code: int = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class RemoteException(BentoMLException):
    """
    Raise when known exceptions happened in remote process
    """

    def __init__(self, message: str, payload: BentoMLException = None):
        self.payload = payload
        super().__init__(message)


class InvalidArgument(BentoMLException):
    """
    Raise when BentoML received unexpected/invalid arguments from CLI arguments, HTTP
    Request, or python API function parameters
    """

    error_code = HTTPStatus.BAD_REQUEST


class InternalServerError(BentoMLException):
    """
    Raise when BentoML received valid arguments from CLI arguments, HTTP
    Request, or python API function parameters, but got internal issues while
    processing.
    * Note to BentoML org developers: raise this exception only when exceptions happend
    in the users' code (runner or service) and want to surface it to the user.
    """


class APIDeprecated(BentoMLException):
    """
    Raise when trying to use deprecated APIs of BentoML
    """


class BadInput(InvalidArgument):
    """Raise when API server receiving bad input request"""

    error_code = HTTPStatus.BAD_REQUEST


class NotFound(BentoMLException):
    """
    Raise when specified resource or name not found
    """

    error_code = HTTPStatus.NOT_FOUND


class TooManyRequests(BentoMLException):
    """
    Raise when incoming requests exceeds the capacity of a server
    """

    error_code = HTTPStatus.TOO_MANY_REQUESTS


class BentoMLConfigException(BentoMLException):
    """Raise when BentoML is mis-configured or when required configuration is missing"""


class MissingDependencyException(BentoMLException):
    """
    Raise when BentoML component failed to load required dependency - some BentoML
    components has dependency that is optional to the library itself. For example,
    when using SklearnModel, the scikit-learn module is required although
    BentoML does not require scikit-learn to be a dependency when installed
    """


class CLIException(BentoMLException):
    """Raise when CLI encounters an issue"""


class YataiRESTApiClientError(BentoMLException):
    pass
