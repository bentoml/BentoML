from abc import ABCMeta, abstractmethod
from six import add_metaclass


@add_metaclass(ABCMeta)
class RequestHandler():

    @staticmethod
    @abstractmethod
    def handle_request(request, func):
        pass


@add_metaclass(ABCMeta)
class CliHandler():

    @staticmethod
    @abstractmethod
    def handle_cli(options, func):
        pass
