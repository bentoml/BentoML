import os
import json
import uuid
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import bentoml
from bentoml.server import BentoModelApiServer

BASE_TEST_PATH = "/tmp/bentoml-test"


class MyFakeModel(object):

    def predict(self, df):
        df['age'] = df['age'].add(5)
        return df


class RestServiceTestModel(bentoml.BentoModel):
    """
    My RestServiceTestModel packaging with BentoML
    """

    def config(self, artifacts, env):
        artifacts.add(bentoml.artifacts.PickleArtifact('fake_model'))

    @bentoml.api(bentoml.handlers.DataframeHandler, options={'input_columns_require': ['age']})
    def predict(self, df):
        """
        predict expects dataframe as input
        """
        return self.artifacts.fake_model.predict(df)


def create_rest_server():
    fake_model = MyFakeModel()
    sm = RestServiceTestModel(fake_model=fake_model)
    version = "test_" + uuid.uuid4().hex
    sm.save(BASE_TEST_PATH, version=version)

    model_path = os.path.join(BASE_TEST_PATH, 'RestServiceTestModel', version)
    model_service = bentoml.load(model_path)
    model_service.load()

    rest_server = BentoModelApiServer('test_rest_server', model_service, 5000)
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
        response_data = json.loads(response.data)
    assert 15 == response_data[0]['age']
