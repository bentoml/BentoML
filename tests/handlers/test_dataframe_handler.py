import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from bentoml import BentoService, api, artifacts  # noqa: E402
from bentoml.artifact import PickleArtifact  # noqa: E402
from bentoml.handlers import DataframeHandler  # noqa: E402

class TestDataframeModel(object):

    def predict(self, df):
        return df['name'][0]

@artifacts([PickleArtifact('test_clf')])
class DataframeHandlerModel(BentoService):

    @api(DataframeHandler)
    def predict(self, df):
        return self.artifacts.test_clf.predict(df)


def test_dataframe_handler(capsys, tmpdir):
    test_content = """
    [
      {
        "name": "john",
        "game": "mario",
        "city": "sf"
      }
    ]
    """
    test_model = TestDataframeModel()
    ms = DataframeHandlerModel.pack(test_clf=test_model)
    api = ms.get_service_apis()[0]

    import json
    json_file = tmpdir.join('test.json')
    with open(json_file, 'w') as f:
        f.write(test_content)

    test_args = ['--input={}'.format(json_file)]
    api.handle_cli(test_args)
    out, err = capsys.readouterr()
    assert out.strip().endswith('john')
