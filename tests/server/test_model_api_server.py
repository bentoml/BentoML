import os
import json
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import bentoml
from bentoml.server import BentoAPIServer
from bentoml.artifact import PickleArtifact
from tests.utils import generate_fake_dataframe_model


def create_rest_server():
    saved_path = generate_fake_dataframe_model()
    model_service = bentoml.load(saved_path)

    rest_server = BentoAPIServer('test_rest_server', model_service, 5000)
    return rest_server


def test_api_function_route():
    rest_server = create_rest_server()
    test_client = rest_server.app.test_client()

    list = []
    for rule in rest_server.app.url_map.iter_rules():
        list.append(rule.endpoint)

    assert 'predict' in list
    data = [{'age': 10}]

    response = test_client.post('/predict', data=json.dumps(data), content_type='application/json')

    if sys.version_info.major < 3:
        import ast
        loaded_json = json.loads(response.data)
        response_data = ast.literal_eval(str(response.data))
    else:
        print(response.data)
        response_data = json.loads(response.data)
    assert 15 == response_data[0]['age']
