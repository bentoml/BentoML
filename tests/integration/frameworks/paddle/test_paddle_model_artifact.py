import random

import numpy as np
import paddle
import paddle.nn as nn
import pandas as pd
import pytest

from bentoml.paddle import PaddlePaddleModel
from tests._internal.frameworks.paddle_utils import LinearModel, test_df
from tests._internal.helpers import assert_have_file_extension

BATCH_SIZE = 8
EPOCH_NUM = 5


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    paddle.framework.random._manual_program_seed(seed)


def predict_df(predictor: paddle.inference.Predictor, df: pd.DataFrame) -> np.ndarray:
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
    set_random_seed(1994)
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


@pytest.fixture()
def create_paddle_predictor(
    train_paddle_model, tmp_path_factory
) -> "paddle.inference.Predictor":
    # Predictor init requires the path of saved model
    tmp_path = str(tmp_path_factory.mktemp("paddle_predictor"))
    paddle.jit.save(train_paddle_model, tmp_path)

    config = paddle.inference.Config(tmp_path + ".pdmodel", tmp_path + ".pdiparams")
    config.enable_memory_optim()
    return paddle.inference.create_predictor(config)


def test_paddle_save_load(tmpdir, train_paddle_model, create_paddle_predictor):
    PaddlePaddleModel(train_paddle_model).save(tmpdir)
    assert_have_file_extension(tmpdir, ".pdmodel")
    paddle_loaded: nn.Layer = PaddlePaddleModel.load(tmpdir)
    compare = predict_df(create_paddle_predictor, test_df) == predict_df(
        paddle_loaded, test_df
    )
    assert compare.all()


def test_paddle_load_custom_conf(train_paddle_model, tmp_path_factory):
    tmp_path = str(tmp_path_factory.mktemp("predictor"))
    paddle.jit.save(train_paddle_model, tmp_path)
    conf = paddle.inference.Config(tmp_path + ".pdmodel", tmp_path + ".pdiparams")
    conf.enable_memory_optim()
    conf.set_cpu_math_library_num_threads(1)
    paddle.set_device("cpu")
    loaded_with_customs: nn.Layer = PaddlePaddleModel.load(tmp_path, config=conf)
    assert predict_df(loaded_with_customs, test_df) == np.array(
        [[0.68022454]], dtype=np.float32
    )
