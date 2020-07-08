import mimetypes
import string
import random

from werkzeug import Request

from bentoml.adapters.multi_image_input import MultiImageInput
from bentoml.marshal.utils import SimpleRequest


def generate_multipart_body(image_file):
    _BOUNDARY_CHARS = string.digits + string.ascii_letters

    boundary = ''.join(random.choice(_BOUNDARY_CHARS) for i in range(30))
    lines = []

    image = {"filename": "image.jpg", "content": open(image_file, "rb").read()}
    files = {"imageX": image, "imageY": image}
    for name, value in files.items():
        filename = value['filename']
        mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        lines.extend(
            (
                bytes('--{0}'.format(boundary), 'utf8'),
                bytes(
                    'Content-Disposition: form-data; name="{0}"; filename="{1}"'.format(
                        name.replace('"', '\\"'), filename.replace('"', '\\"')
                    ), 'utf8'),
                bytes('Content-Type: {0}'.format(mime_type), 'utf8'),
                b'',
                value['content'],
            )
        )

    lines.extend((bytes('--{0}--'.format(boundary), 'utf8'), b'',))
    body = b'\r\n'.join(lines)

    headers = {
        'Content-Type': 'multipart/form-data; boundary={0}'.format(boundary),
        'Content-Length': len(body),
    }
    return body, headers


def predict(requests):
    return [(request['imageX'].shape, request['imageY'].shape) for request in requests]


def test_multi_image_input_cli(capsys, img_file):
    adapter = MultiImageInput(input_names=("imageX", "imageY"), is_batch_input=True)

    test_args = ['--imageX', img_file, '--imageY', img_file]
    adapter.handle_cli(test_args, predict)
    out, _ = capsys.readouterr()
    assert out.strip() == '((10, 10, 3), (10, 10, 3))'


def test_multi_image_input_request(img_file):
    adapter = MultiImageInput(("imageX", "imageY"))

    multipart_data, headers = generate_multipart_body(img_file)
    request = Request.from_values(
        data=multipart_data,
        content_type=headers['Content-Type'],
        content_length=headers['Content-Length'],
    )

    response = adapter.handle_request(request, predict)
    assert response.status_code == 200
    assert response.data == b'[[10, 10, 3], [10, 10, 3]]'


def test_multi_image_batch_input(img_file):
    adapter = MultiImageInput(("imageX", "imageY"), is_batch_input=True)

    multipart_data, headers = generate_multipart_body(img_file)
    request = SimpleRequest.from_flask_request(
        Request.from_values(
            data=multipart_data,
            content_type=headers['Content-Type'],
            content_length=headers['Content-Length'],
        )
    )

    responses = adapter.handle_batch_request([request] * 5, predict)
    for response in responses:
        assert response.status == 200
        assert response.data == '[[10, 10, 3], [10, 10, 3]]'


def test_multi_image_aws_event(img_file):
    adapter = MultiImageInput(("imageX", "imageY"), is_batch_input=True)

    multipart_data, headers = generate_multipart_body(img_file)
    aws_lambda_event = {
        "body": multipart_data,
        "headers": headers
    }

    aws_result = adapter.handle_aws_lambda_event(aws_lambda_event, predict)
    assert aws_result["statusCode"] == 200
    assert aws_result["body"] == '[[10, 10, 3], [10, 10, 3]]'
