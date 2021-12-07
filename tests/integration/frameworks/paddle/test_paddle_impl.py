import numpy as np
import paddle
import pytest
import paddle.nn as nn
from paddle.static import InputSpec

import bentoml.paddle
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.paddle_utils import test_df
from tests.utils.frameworks.paddle_utils import IN_FEATURES

from .conftest import predict_df


@pytest.mark.parametrize(
    "input_spec", [None, [InputSpec(shape=[IN_FEATURES], dtype="float32")]]
)
def test_paddle_save_load(train_paddle_model, input_spec, modelstore):
    tag = bentoml.paddle.save(
        "linear_model",
        train_paddle_model,
        model_store=modelstore,
        input_spec=input_spec,
    )
    info = modelstore.get(tag)
    assert_have_file_extension(info.path, ".pdmodel")
    loaded = bentoml.paddle.load(tag, model_store=modelstore)
    assert predict_df(loaded, test_df) == np.array([[0.9003858]], dtype=np.float32)


def test_paddle_load_custom_conf(train_paddle_model, modelstore):
    tag = bentoml.paddle.save(
        "linear_model", train_paddle_model, model_store=modelstore
    )
    info = modelstore.get(tag)
    conf = paddle.inference.Config(
        info.path + "/saved_model.pdmodel", info.path + "/saved_model.pdiparams"
    )
    conf.enable_memory_optim()
    conf.set_cpu_math_library_num_threads(1)
    paddle.set_device("cpu")
    loaded_with_customs: nn.Layer = bentoml.paddle.load(
        tag, config=conf, model_store=modelstore
    )
    assert predict_df(loaded_with_customs, test_df) == np.array(
        [[0.9003858]], dtype=np.float32
    )
