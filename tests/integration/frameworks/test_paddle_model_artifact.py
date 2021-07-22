import os
import typing as t

import numpy as np
import paddle
import paddle.nn as nn
import paddlehub
import pandas as pd
import pytest
from paddle.static import InputSpec

from bentoml.paddle import PaddleHubModel, PaddlePaddleModel

BATCH_SIZE = 8
BATCH_NUM = 4
EPOCH_NUM = 5

IN_FEATURES = 13
OUT_FEATURES = 1

test_df = pd.DataFrame(
    [
        [
            -0.0405441,
            0.06636364,
            -0.32356227,
            -0.06916996,
            -0.03435197,
            0.05563625,
            -0.03475696,
            0.02682186,
            -0.37171335,
            -0.21419304,
            -0.33569506,
            0.10143217,
            -0.21172912,
        ]
    ]
)

test_text: t.List[str] = ["味道不错，确实不算太辣，适合不能吃辣的人。就在长江边上，抬头就能看到长江的风景。鸭肠、黄鳝都比较新鲜。"]


def predict_df(predictor: paddle.inference.Predictor, df: pd.DataFrame):
    input_data = df.to_numpy().astype(np.float32)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])
    input_handle.reshape(input_data.shape)
    input_handle.copy_from_cpu(input_data)

    predictor.run()

    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    return output_handle.copy_to_cpu()


class Model(nn.Layer):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(IN_FEATURES, OUT_FEATURES)

    @paddle.jit.to_static(input_spec=[InputSpec(shape=[IN_FEATURES], dtype='float32')])
    def forward(self, x):
        return self.fc(x)


@pytest.fixture(scope='session')
def train_paddle_model() -> "Model":
    model = Model()
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
    tmpdir = str(tmp_path_factory.mktemp('paddle_predictor'))
    paddle.jit.save(train_paddle_model, tmpdir)

    config = paddle.inference.Config(tmpdir + '.pdmodel', tmpdir + '.pdiparams')
    config.enable_memory_optim()
    return paddle.inference.create_predictor(config)


def test_paddle_save_load(tmpdir, train_paddle_model, create_paddle_predictor):
    PaddlePaddleModel(train_paddle_model).save(tmpdir)
    assert os.path.exists(PaddlePaddleModel.get_path(tmpdir, '.pdmodel'))
    paddle_loaded: nn.Layer = PaddlePaddleModel.load(tmpdir)
    assert (
        predict_df(create_paddle_predictor, test_df).shape
        == predict_df(paddle_loaded, test_df).shape
    )


def test_paddlehub_save_load(tmpdir):
    senta = paddlehub.Module(name="senta_bilstm")

    print(senta.sentiment_classify(texts=test_text))
    PaddleHubModel(senta).save(tmpdir)
    assert os.path.exists(os.path.join(tmpdir, 'model.pdmodel'))

    paddlehub_loaded = PaddleHubModel.load(tmpdir)
    assert (
        senta.sentiment_classify(texts=test_text)[0]['positive_probs']
        == paddlehub_loaded.sentiment_classify(texts=test_text)[0]['positive_probs']
    )
