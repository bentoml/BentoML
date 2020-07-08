from werkzeug import Request

from bentoml.adapters.multi_image_input import MultiImageInput
from bentoml.marshal.utils import SimpleRequest


def predict(requests):
    return [(request['imageX'].shape, request['imageY'].shape) for request in requests]


def test_multi_image_input_cli(capsys, img_file):
    adapter = MultiImageInput(input_names=("imageX", "imageY"), is_batch_input=True)

    test_args = ['--imageX', img_file, '--imageY', img_file]
    adapter.handle_cli(test_args, predict)
    out, _ = capsys.readouterr()
    assert out.strip() == '((10, 10, 3), (10, 10, 3))'


def test_multi_image_input_request():
    adapter = MultiImageInput(("imageX", "imageY"))

    multipart_data = open("tests/multipart", 'rb').read()
    request = Request.from_values(
        data=multipart_data,
        content_type="multipart/form-data; boundary"
                     "=---------------------------309036539235529840702074443043",
        content_length=len(multipart_data)
    )

    response = adapter.handle_request(request, predict)
    assert response.status_code == 200
    assert response.data == b'[[779, 935, 3], [779, 935, 3]]'


def test_multi_image_batch_input():
    adapter = MultiImageInput(("imageX", "imageY"), is_batch_input=True)

    multipart_data = open("tests/multipart", 'rb').read()
    request = SimpleRequest.from_flask_request(Request.from_values(
        data=multipart_data,
        content_type="multipart/form-data; boundary"
                     "=---------------------------309036539235529840702074443043",
        content_length=len(multipart_data)
    ))

    responses = adapter.handle_batch_request([request] * 5, predict)
    for response in responses:
        assert response.status == 200
        assert response.data == '[[779, 935, 3], [779, 935, 3]]'
