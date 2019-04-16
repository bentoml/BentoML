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


class TestImageModel(object):

    def predict(self, image_ndarray):
        return image_ndarray.shape


@artifacts([PickleArtifact('clf')])
class ImageHandlerModel(BentoService):

    @api(ImageHandler)
    def predict(self, input_data):
        return self.artifacts.clf.predict(input_data)


BASE_TEST_PATH = "/tmp/bentoml-test"


def test_image_handler(capsys):
    test_model = TestImageModel()
    ms = ImageHandlerModel.pack(clf=test_model)
    api = ms.get_service_apis()[0]

    test_args = ['--input=./tests/test_image_1.jpg']
    api.handle_cli(test_args)
    out, err = capsys.readouterr()

    assert out.strip().endswith('(360, 480, 3)')
