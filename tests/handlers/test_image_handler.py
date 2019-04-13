import os
import sys

try:
    from bentoml import BentoService, api, artifacts
    from bentoml.artifact import PickleArtifact
    from bentoml.handlers import ImageHandler
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from bentoml import BentoService, api, artifacts
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
    api.handle_cli(fake_args)
    out, err = capsys.readouterr()

    assert out.strip().endswith('(360, 480, 3)')
