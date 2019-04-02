import json
import pandas as pd
from flask import Response, make_response
from bentoml.handlers import CliHandler, RequestHandler


class DataframeHandler(RequestHandler, CliHandler):
    """
    Create Data frame handler.  Dataframe handler will take inputs from rest request
    or cli options and return response for REST or stdout for CLI
    """

    @staticmethod
    def handle_request(request, func):
        if request.content_type == 'application/json':
            df = pd.read_json(request.data.decode('utf-8'))
        elif request.content_type == 'text/csv':
            df = pd.read_csv(request.data.decode('utf-8'))
        else:
            return make_response(400)

        output = func(df)

        if isinstance(output, pd.DataFrame):
            result = output.to_json()
        else:
            result = json.dumps(output)

        response = Response(response=result, status=200, mimetype="application/json")
        return response

    @staticmethod
    def handle_cli(options, func):
        raise NotImplementedError
