from bentoml.handlers.base_handlers import RequestHandler, CliHandler


class TensorflowTensorHandler(RequestHandler, CliHandler):
    """
    Tensor handlers for Tensorflow models
    """

    @staticmethod
    def handle_request(request, func):
        raise NotImplementedError

    @staticmethod
    def handle_cli(options, func):
        raise NotImplementedError
