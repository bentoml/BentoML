import base64

from bentoml import BentoService, api
from bentoml.handlers import ImageHandler



class ImageHandlerModel(BentoService):
    @api(ImageHandler)
    def predict(self, image):
        return image.shape


def test_image_handler(capsys, tmpdir):
    ms = ImageHandlerModel()
    api = ms.get_service_apis()[0]

    import cv2
    import numpy as np

    img_file = tmpdir.join("img.png")
    cv2.imwrite(str(img_file), np.zeros((10, 10)))

    test_args = ["--input={}".format(img_file)]
    api.handle_cli(test_args)
    out, err = capsys.readouterr()

    assert out.strip().endswith("(10, 10, 3)")

    with open(str(img_file), "rb") as imageFile:
        content = imageFile.read()
        try:
            imageStr = base64.encodebytes(content)
        except AttributeError:
            imageStr = base64.encodestring(str(img_file))
    aws_lambda_event = {"body": imageStr, "headers": {"Content-Type": "images/png"}}

    aws_result = api.handle_aws_lambda_event(aws_lambda_event)
    assert aws_result["statusCode"] == 200
    assert aws_result["body"] == "[10, 10, 3]"
