import random

import numpy as np
import paddle
import pandas as pd
import pytest
import paddle.nn as nn

from tests.utils.frameworks.paddle_utils import LinearModel

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
