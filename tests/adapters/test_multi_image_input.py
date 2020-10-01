# pylint: disable=redefined-outer-name
import io

import pytest

from bentoml.adapters import MultiImageInput
from bentoml.types import InferenceTask


@pytest.fixture()
def input_adapter():
    return MultiImageInput(input_names=("image1", "image2"))


@pytest.fixture()
def img_bytes(img_file):
    with open(img_file, "rb") as f:
        return f.read()


@pytest.fixture()
def gen_img_io(img_bytes):
    def _(name="test.jpg"):
        img_io = io.BytesIO(img_bytes)
        img_io.name = name
        return img_io

    return _


@pytest.fixture()
def verify_args():
    def _(args):
        assert args[0]
        assert args[1]

        for img1, img2 in zip(*args):
            assert img1.shape == (10, 10, 3)
            assert img2.shape == (10, 10, 3)

    return _


def test_multi_image_input_extract_args(input_adapter, gen_img_io, verify_args):
    task = InferenceTask(data=(gen_img_io(), gen_img_io()))
    args = input_adapter.extract_user_func_args([task])
    verify_args(args)


def test_multi_image_input_extract_args_wrong_extension(input_adapter, gen_img_io):
    task = InferenceTask(data=(gen_img_io("test.custom"), gen_img_io("test.jpg")))
    args = input_adapter.extract_user_func_args([task])
    for _ in zip(*args):
        pass
    assert task.is_discarded


def test_multi_image_input_extract_args_custom_extension(gen_img_io, verify_args):
    input_adapter = MultiImageInput(accept_image_formats=[".custom", ".jpg"])
    task = InferenceTask(data=(gen_img_io("test.custom"), gen_img_io("test.jpg")))
    args = input_adapter.extract_user_func_args([task])
    assert not task.is_discarded
    verify_args(args)


def test_multi_image_input_extract_args_missing_image(input_adapter, gen_img_io):
    task = InferenceTask(data=(None, gen_img_io()))
    args = input_adapter.extract_user_func_args([task])

    assert not args[0]
    for _ in zip(*args):
        pass
    assert task.is_discarded


def test_anno_image_input_check_config(input_adapter):
    config = input_adapter.config
    assert isinstance(config["accept_image_formats"], list) and isinstance(
        config["pilmode"], str
    )


def test_anno_image_input_check_request_schema(input_adapter):
    assert isinstance(input_adapter.request_schema, dict)


def test_anno_image_input_check_pip_deps(input_adapter):
    assert isinstance(input_adapter.pip_dependencies, list)
