import os
import json
import sys

try:
    import bentoml
    from bentoml.server import BentoAPIServer
    from tests.utils import generate_fake_dataframe_model
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    import bentoml
    from bentoml.server import BentoAPIServer
    from tests.utils import generate_fake_dataframe_model


def create_rest_server():
    saved_path = generate_fake_dataframe_model()
    model_service = bentoml.load(saved_path)

    rest_server = BentoAPIServer(model_service)
    return rest_server


def test_api_function_route():
    rest_server = create_rest_server()
    test_client = rest_server.app.test_client()

    index_list = []
    for rule in rest_server.app.url_map.iter_rules():
        index_list.append(rule.endpoint)

    assert 'predict' in index_list
    data = [{'age': 10}]

    response = test_client.post('/predict', data=json.dumps(data), content_type='application/json')

    response_data = json.loads(response.data)
    assert 15 == response_data[0]['age']
