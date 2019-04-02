from bentoml.handlers.base_handlers import RequestHandler, CliHandler


class ImageHandler(RequestHandler, CliHandler):
    """
    Image handler take input image and process them and return response or stdout.
    """

    @staticmethod
    def handle_request(request, func):
        raise NotImplementedError

    @staticmethod
    def handle_cli(options, func):
        raise NotImplementedError
