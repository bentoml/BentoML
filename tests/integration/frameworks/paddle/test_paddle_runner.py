import numpy as np
import pytest

import bentoml
import paddle.inference

from .test_paddle_impl import test_df


def test_paddlepaddle_load_runner(train_paddle_model):
    tag = bentoml.paddle.save("linear_model", train_paddle_model)
    info = bentoml.models.get(tag)
    runner = bentoml.paddle.load_runner(tag)

    assert info.tag in runner.required_models
    assert runner.num_replica == 1

    input_data = test_df.to_numpy().astype(np.float32)
    assert runner.run_batch(input_data) == [np.array([0.90038574], dtype=np.float32)]
    assert isinstance(runner._model, paddle.inference.Predictor)


def test_paddlepaddle_runner_from_paddlehub():
    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
    tag = bentoml.paddle.import_from_paddlehub("senta_bilstm")
    runner = bentoml.paddle.load_runner(tag, infer_api_callback="sentiment_classify")
    results = runner.run_batch(None, texts=test_text, use_gpu=False, batch_size=1)
    assert results[0]["positive_probs"] == 0.9407
    assert results[1]["positive_probs"] == 0.02


@pytest.mark.gpus
def test_paddlepaddle_load_runner_gpu(train_paddle_model):
    """
    tag = bentoml.paddle.save("linear_model", train_paddle_model)
    info = bentoml.models.get(tag)
    runner = bentoml.paddle.load_runner(
        tag,
        enable_gpu=True,
        device="gpu:0",
    )

    assert info.tag in runner.required_models
    assert runner.num_replica == bentoml.paddle.device_count()

    input_data = test_df.to_numpy().astype(np.float32)
    _ = runner.run_batch(input_data)
    assert isinstance(runner._model, paddle.inference.Predictor)
    assert runner._runner_config.use_gpu() is True
    """
