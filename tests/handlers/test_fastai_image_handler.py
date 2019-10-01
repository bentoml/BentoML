import sys

import bentoml
from bentoml.handlers import FastaiImageHandler


def test_fastai_image_handler(capsys, tmpdir):
    if sys.version_info < (3, 6):
        # fast ai is required 3.6 or higher.
        assert True
    else:
        class ImageHandlerModelForFastai(bentoml.BentoService):
            @bentoml.api(FastaiImageHandler)
            def predict(self, image):
                return list(image.shape)

        ms = ImageHandlerModelForFastai()

        import cv2
        import numpy as np

        img_file = tmpdir.join("img.png")
        cv2.imwrite(str(img_file), np.zeros((10, 10)))
        api = ms.get_service_apis()[0]
        test_args = ["--input={}".format(img_file)]
        api.handle_cli(test_args)
        out, err = capsys.readouterr()
        assert out.strip() == '[3, 10, 10]'
