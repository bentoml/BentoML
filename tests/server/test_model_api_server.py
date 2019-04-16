import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from bentoml.server import BentoAPIServer  # noqa: E402


def test_api_function_route(bento_service):
    rest_server = BentoAPIServer(bento_service)
    test_client = rest_server.app.test_client()

    index_list = []
    for rule in rest_server.app.url_map.iter_rules():
        index_list.append(rule.endpoint)

    assert 'predict' in index_list
    data = [{'age': 10}]

    response = test_client.post('/predict', data=json.dumps(data), content_type='application/json')

    response_data = json.loads(response.data)
    assert 15 == response_data[0]['age']
