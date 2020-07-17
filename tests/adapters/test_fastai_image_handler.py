import pytest

import bentoml
from bentoml.adapters import FastaiImageInput

pytestmark = pytest.mark.skip("skipping entire test file to debug travis build issue")


def test_fastai_image_input(capsys, tmpdir):
    class ImageInputModelForFastai(bentoml.BentoService):
        @bentoml.api(input=FastaiImageInput())
        def predict(self, image):
            return list(image.shape)

    ms = ImageInputModelForFastai()

    import imageio
    import numpy as np

    img_file = tmpdir.join("img.png")
    imageio.imwrite(str(img_file), np.zeros((10, 10)))
    api = ms.inference_apis[0]
    test_args = ["--input={}".format(img_file)]
    api.handle_cli(test_args)
    out, _ = capsys.readouterr()
    assert out.strip() == '[3, 10, 10]'
