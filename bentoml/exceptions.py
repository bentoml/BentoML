from http import HTTPStatus


class BentoMLException(Exception):
    """
    Base class for all BentoML's errors.
    Each custom exception should be derived from this class
    """

    error_code: int = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, msg: str):
        super().__init__(msg)


class RemoteException(BentoMLException):
    """
    Raise when known exceptions happened in remote process
    """


class InvalidArgument(BentoMLException):
    """
    Raise when BentoML received unexpected/invalid arguments from CLI arguments, HTTP
    Request, or python API function parameters
    """

    error_code = HTTPStatus.BAD_REQUEST


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