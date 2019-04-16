import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from bentoml import BentoService, api, artifacts  # noqa: E402
from bentoml.artifact import PickleArtifact  # noqa: E402
from bentoml.handlers import ImageHandler  # noqa: E402


class TestImageModel(object):

    def predict(self, image_ndarray):
        return image_ndarray.shape


@artifacts([PickleArtifact('clf')])
class ImageHandlerModel(BentoService):

    @api(ImageHandler)
    def predict(self, input_data):
        return self.artifacts.clf.predict(input_data)


def test_image_handler(capsys, tmpdir):
    test_model = TestImageModel()
    ms = ImageHandlerModel.pack(clf=test_model)
    api = ms.get_service_apis()[0]

    import cv2
    import numpy as np
    img_file = tmpdir.join('img.png')
    cv2.imwrite(str(img_file), np.zeros((10, 10)))

    test_args = ['--input={}'.format(img_file)]
    api.handle_cli(test_args)
    out, err = capsys.readouterr()

    assert out.strip().endswith('(10, 10, 3)')
