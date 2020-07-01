from bentoml.adapters.multi_image_input import MultiImageInput


def predict(requests):
    return [(request['imageX'].shape, request['imageY'].shape) for request in requests]


def test_multi_image_input_cli(capsys, img_file):
    adapter = MultiImageInput(input_names=("imageX", "imageY"))

    test_args = ['--imageX', img_file, '--imageY', img_file]
    adapter.handle_cli(test_args, predict)
    out, _ = capsys.readouterr()
    assert out.strip() == '((10, 10, 3), (10, 10, 3))'
