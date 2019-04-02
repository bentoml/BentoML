import json
from flask import Response, make_response

from bentoml.handlers.base_handlers import RequestHandler, CliHandler


class JsonHandler(RequestHandler, CliHandler):
    """
    Json handler take input json str and process them and return response or stdout.
    """

    @staticmethod
    def handle_request(request, func):
        if request.content_type == 'application/json':
            parsed_json = json.loads(request.data.decode('utf-8'))
        else:
            return make_response(400)

        output = func(parsed_json)

        response = Response(response=json.dumps(output), status=200, mimetype="application/json")
        return response

    @staticmethod
    def handle_cli(options, func):
        raise NotImplementedError
