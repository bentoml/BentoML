from bentoml.handlers.base_handlers import RequestHandler, CliHandler


class JsonHandler(RequestHandler, CliHandler):
    """
    Json handler take input json str and process them and return response or stdout.
    """

    @staticmethod
    def handle_request(request, func):
        return func(request)

    @staticmethod
    def handle_cli(options, func):
        return func(options)
