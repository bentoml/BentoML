# from flask import Request, Response, make_response, request
# import numpy as np
# import json
from bentoml.handlers.base_handlers import CliHandler, RequestHandler


class PytorchTensorHanlder(RequestHandler, CliHandler):
    """
    Tensor handlers for Pytorch models
    """

    @staticmethod
    def handle_request(request, func):
        return func(request)

    @staticmethod
    def handle_cli(options, func):
        return func(options)
