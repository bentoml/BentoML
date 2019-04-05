import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import bentoml



BASE_TEST_PATH = "/tmp/bentoml-test"


class MyFakeModel(object):
    def predict(self, data):
        data['age'] = data['age'] * 2
        return data


class MyFakeBentoModel(bentoml.BentoModel):
    """
    My Fake Model packaging with BentoML
    """

    def config(self, artifacts, env):
        artifacts.add(bentoml.artifacts.PickleArtifact('fake_model'))

    @bentoml.api(bentoml.handlers.JsonHandler)
    def predict(self, data):
        """
        predict expects dataframe as input
        """
        return self.artifacts.fake_model.predict(data)


def generate_fake_model():
    """
    Generate a fake model, saved it to tmp and return saved_path
    """
    fake_model = MyFakeBentoModel()
    ms = MyFakeBentoModel(fake_model=fake_model)
    saved_path = ms.save(BASE_TEST_PATH)

    return saved_path
