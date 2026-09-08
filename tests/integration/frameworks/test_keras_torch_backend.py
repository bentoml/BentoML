from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.mark.skipif(
    os.environ.get("KERAS_BACKEND", "tensorflow") != "torch",
    reason="This test exercises the Keras torch backend.",
)
def test_keras_torch_save_load():
    import keras

    import bentoml

    backend = keras.config.backend()
    if backend != "torch":
        pytest.skip(f"Keras backend is {backend!r}, not 'torch'.")

    # Build a tiny deterministic model.
    inp = keras.Input(shape=(3,), name="inp")
    out = keras.layers.Dense(1, use_bias=False, kernel_initializer="ones", name="out")(
        inp
    )
    model = keras.Model(inputs=inp, outputs=out)

    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    expected = model.predict(x)

    bento_model = bentoml.keras.save_model("keras_torch_model", model)

    loaded = bentoml.keras.load_model(bento_model.tag)
    np.testing.assert_allclose(loaded.predict(x), expected, rtol=1e-5)

    runner = bento_model.to_runner()
    runner.init_local()
    try:
        result = runner.predict.run(x)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    finally:
        runner.destroy()


@pytest.mark.skipif(
    os.environ.get("KERAS_BACKEND", "tensorflow") != "torch",
    reason="This test exercises the Keras torch backend.",
)
def test_keras_torch_load_with_tf_style_device():
    import keras

    import bentoml

    backend = keras.config.backend()
    if backend != "torch":
        pytest.skip(f"Keras backend is {backend!r}, not 'torch'.")

    inp = keras.Input(shape=(3,), name="inp")
    out = keras.layers.Dense(1, use_bias=False, kernel_initializer="ones", name="out")(
        inp
    )
    model = keras.Model(inputs=inp, outputs=out)

    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    expected = model.predict(x)

    bento_model = bentoml.keras.save_model("keras_torch_model_tf_device", model)

    # TensorFlow-style device strings should be accepted and normalized.
    loaded = bentoml.keras.load_model(bento_model.tag, device_name="/device:CPU:0")
    np.testing.assert_allclose(loaded.predict(x), expected, rtol=1e-5)

    runner = bento_model.to_runner()
    runner.init_local()
    try:
        result = runner.predict.run(x)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    finally:
        runner.destroy()


@pytest.mark.skipif(
    os.environ.get("KERAS_BACKEND", "tensorflow") != "torch",
    reason="This test exercises the Keras torch backend.",
)
def test_keras_torch_runner_accepts_list_input():
    import keras

    import bentoml

    backend = keras.config.backend()
    if backend != "torch":
        pytest.skip(f"Keras backend is {backend!r}, not 'torch'.")

    inp = keras.Input(shape=(3,), name="inp")
    out = keras.layers.Dense(1, use_bias=False, kernel_initializer="ones", name="out")(
        inp
    )
    model = keras.Model(inputs=inp, outputs=out)

    bento_model = bentoml.keras.save_model("keras_torch_model_list_input", model)

    x_list = [[1.0, 2.0, 3.0]]
    x_tuple = ([1.0, 2.0, 3.0],)
    expected = model.predict(np.array(x_list, dtype=np.float32))

    runner = bento_model.to_runner()
    runner.init_local()
    try:
        np.testing.assert_allclose(runner.predict.run(x_list), expected, rtol=1e-5)
        np.testing.assert_allclose(runner.predict.run(x_tuple), expected, rtol=1e-5)
    finally:
        runner.destroy()


@pytest.mark.skipif(
    os.environ.get("KERAS_BACKEND", "tensorflow") != "torch",
    reason="This test exercises the Keras torch backend.",
)
def test_keras_torch_load_with_wrong_backend_raises():
    import keras

    import bentoml
    from bentoml.exceptions import BentoMLException

    backend = keras.config.backend()
    if backend != "torch":
        pytest.skip(f"Keras backend is {backend!r}, not 'torch'.")

    inp = keras.Input(shape=(3,), name="inp")
    out = keras.layers.Dense(1, use_bias=False, kernel_initializer="ones", name="out")(
        inp
    )
    model = keras.Model(inputs=inp, outputs=out)

    bento_model = bentoml.keras.save_model("keras_torch_model_wrong_backend", model)

    # Simulate a mismatch by patching the saved backend to "tensorflow".
    bento_model.info.context.framework_versions["backend"] = "tensorflow"

    with pytest.raises(
        BentoMLException,
        match="was saved with backend 'tensorflow'",
    ):
        bentoml.keras.load_model(bento_model)
