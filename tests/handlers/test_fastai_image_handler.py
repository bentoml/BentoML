import pytest

import bentoml
from bentoml.handlers import FastaiImageHandler

pytestmark = pytest.mark.skip("skipping entire test file to debug travis build issue")


def test_fastai_image_handler(capsys, tmpdir):
    class ImageHandlerModelForFastai(bentoml.BentoService):
        @bentoml.api(FastaiImageHandler)
        def predict(self, image):
            return list(image.shape)

    ms = ImageHandlerModelForFastai()

    import imageio
    import numpy as np

    img_file = tmpdir.join("img.png")
    imageio.imwrite(str(img_file), np.zeros((10, 10)))
    api = ms.get_service_apis()[0]
    test_args = ["--input={}".format(img_file)]
    api.handle_cli(test_args)
    out, err = capsys.readouterr()
    assert out.strip() == '[3, 10, 10]'
