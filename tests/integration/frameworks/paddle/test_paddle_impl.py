import random

import numpy as np
import paddle
import pandas as pd
import pytest
import paddle.nn as nn
from paddle.static import InputSpec

import bentoml.paddle
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.paddle_utils import test_df, IN_FEATURES, LinearModel

BATCH_SIZE = 8
EPOCH_NUM = 5
SEED = 1994


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    paddle.framework.random._manual_program_seed(seed)


def predict_df(predictor: paddle.inference.Predictor, df: pd.DataFrame) -> np.ndarray:
    set_random_seed(SEED)
    input_data = df.to_numpy().astype(np.float32)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle.reshape(input_data.shape)
    input_handle.copy_from_cpu(input_data)

    predictor.run()

    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    return np.asarray(output_handle.copy_to_cpu())


@pytest.fixture(scope="session")
def train_paddle_model() -> "LinearModel":
    set_random_seed(SEED)
    model = LinearModel()
    loss = nn.MSELoss()
    adam = paddle.optimizer.Adam(parameters=model.parameters())

    train_data = paddle.text.datasets.UCIHousing(mode="train")

    loader = paddle.io.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2
    )

    model.train()
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (feature, label) in enumerate(loader()):
            out = model(feature)
            loss_fn = loss(out, label)
            loss_fn.backward()
            adam.step()
            adam.clear_grad()
    return model


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
    set_random_seed(SEED)
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
