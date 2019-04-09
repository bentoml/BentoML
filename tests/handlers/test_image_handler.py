import os
import io
import argparse
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pytest

from bentoml import BentoService, load, api, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.handlers import ImageHandler


class FakeImageModel(object):
    def predict(self, image_ndarray):
        return image_ndarray.shape


@artifacts([
    PickleArtifact('clf')
])
class ImageHandlerModel(BentoService):
    @api(ImageHandler)
    def predict(self, input_data):
        return self.artifacts.clf.predict(input_data)


BASE_TEST_PATH = "/tmp/bentoml-test"


def test_image_handler(capsys):
    fake_model = FakeImageModel()
    ms = ImageHandlerModel.pack(clf=fake_model)
    api = ms.get_service_apis()[0]

    fake_args = ['--input=./tests/test_image_1.jpg']
    api.handler.handle_cli(fake_args, api.func, {})
    out, err = capsys.readouterr()

    assert out.endswith('[360, 480, 3]')
