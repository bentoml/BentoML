import sys

from bentoml import BentoService, api, artifacts
from bentoml.artifact import PickleArtifact
from bentoml.handlers import FastaiImageHandler


class TestImageModel(object):
    def predict(self, image_ndarray):
        return image_ndarray.shape


@artifacts([PickleArtifact("clf")])
class ImageHandlerModelForFastai(BentoService):
    @api(FastaiImageHandler)
    def predict(self, input_data):
        return type(input_data).__name__


def test_fastai_image_handler(capsys, tmpdir):
    if sys.version_info.major < 3 or sys.version_info.minor < 6:
        # fast ai is required 3.6 or higher.
        assert True
    else:
        test_model = TestImageModel()
        ms = ImageHandlerModelForFastai.pack(clf=test_model)

        import cv2
        import numpy as np

        img_file = tmpdir.join("img.png")
        cv2.imwrite(str(img_file), np.zeros((10, 10)))
        api = ms.get_service_apis()[0]
        test_args = ["--input={}".format(img_file)]
        api.handle_cli(test_args)
        out, err = capsys.readouterr()
        assert out.strip() == "Image"
