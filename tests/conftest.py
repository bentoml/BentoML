import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import bentoml  # noqa: E402
from bentoml.artifact import PickleArtifact  # noqa: E402


class TestModel(object):

    def predict(self, df):
        df['age'] = df['age'].add(5)
        return df


@bentoml.artifacts([PickleArtifact('model')])
@bentoml.env()
class TestBentoService(bentoml.BentoService):
    """
    My RestServiceTestModel packaging with BentoML
    """

    @bentoml.api(bentoml.handlers.DataframeHandler, input_columns=['age'])
    def predict(self, df):
        """
        predict expects dataframe as input
        """
        return self.artifacts.model.predict(df)


@pytest.fixture()
def bento_service():
    """
    Create a new TestBentoService
    """
    test_model = TestModel()
    return TestBentoService.pack(model=test_model)


@pytest.fixture()
def bento_archive_path(bento_service, tmpdir):
    """
    Create a new TestBentoService, saved it to tmpdir, and return full saved_path
    """
    saved_path = bento_service.save(str(tmpdir))
    return saved_path
