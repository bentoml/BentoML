class ManagerException(Exception):
    """A generic exception for everything under manager CLI"""


class ManagerLoginFailed(ManagerException):
    """Raised when failed to login to a registry"""


class ManagerBuildFailed(ManagerException):
    """Raised when failed to build releases"""
