import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import bentoml
from bentoml.artifact import PickleArtifact



BASE_TEST_PATH = "/tmp/bentoml-test"



class MyFakeModel(object):
    def predict(self, df):
        df['age'] = df['age'].add(5)
        return df


@bentoml.artifacts([
    PickleArtifact('fake_model')
])
@bentoml.env()
class MyFakeBentoModel(bentoml.BentoService):
    """
    My RestServiceTestModel packaging with BentoML
    """

    @bentoml.api(bentoml.handlers.DataframeHandler, options={'input_columns_require': ['age']})
    def predict(self, df):
        """
        predict expects dataframe as input
        """
        return self.artifacts.fake_model.predict(df)


def generate_fake_dataframe_model():
    """
    Generate a fake model, saved it to tmp and return saved_path
    """
    fake_model = MyFakeModel()
    ms = MyFakeBentoModel.pack(fake_model=fake_model)
    saved_path = ms.save(BASE_TEST_PATH)

    return saved_path
