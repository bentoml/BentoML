from typing import Any
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tensorflow as tf

import bentoml
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.tensorflow_utils import NativeModel
from tests.utils.frameworks.tensorflow_utils import MultiInputModel
from tests.utils.frameworks.tensorflow_utils import NativeRaggedModel
from tests.utils.frameworks.tensorflow_utils import KerasSequentialModel
from bentoml._internal.frameworks.tensorflow_v2 import TensorflowTensorContainer

if TYPE_CHECKING:
    from bentoml._internal.external_typing import tensorflow as tf_ext

MODEL_NAME = __name__.split(".")[-1]

test_data = [[1.1, 2.2]]
test_tensor: "tf_ext.TensorLike" = tf.constant(test_data)

native_data = [[1, 2, 3, 4, 5]]
native_tensor: "tf_ext.TensorLike" = tf.constant(native_data, dtype=tf.float64)

ragged_data = [[15], [7, 8], [1, 2, 3, 4, 5]]
ragged_tensor: "tf_ext.TensorLike" = tf.ragged.constant(ragged_data, dtype=tf.float64)


def _model_dunder_call(
    model: "tf_ext.Module", tensor: "tf_ext.TensorLike"
) -> "tf_ext.TensorLike":
    return model(tensor)


def _tensor_equal(t1: "tf_ext.TensorLike", t2: "tf_ext.TensorLike") -> bool:
    return tf.math.equal(t1, t2).numpy().all()


@pytest.mark.parametrize(
    "mcls, tensor, predict_fn, is_ragged",
    [
        (KerasSequentialModel(), native_tensor, _model_dunder_call, False),
        (NativeModel(), native_tensor, _model_dunder_call, False),
        (NativeRaggedModel(), ragged_tensor, _model_dunder_call, True),
    ],
)
def test_tensorflow_v2_save_load(
    mcls: "tf_ext.Module",
    tensor: "tf_ext.TensorLike",
    predict_fn: Callable[
        ["tf_ext.AutoTrackable", "tf_ext.TensorLike"], "tf_ext.TensorLike"
    ],
    is_ragged: bool,
):

    labels = {"stage": "dev"}

    def custom_f(x: int) -> int:
        return x + 1

    tag = bentoml.tensorflow.save_model(
        MODEL_NAME, mcls, labels=labels, custom_objects={"func": custom_f}
    )
    bentomodel = bentoml.models.get(tag)
    assert_have_file_extension(bentomodel.path, ".pb")
    for k in labels.keys():
        assert labels[k] == bentomodel.info.labels[k]
    assert bentomodel.custom_objects["func"](3) == custom_f(3)

    model = bentoml.tensorflow.load_model(MODEL_NAME)
    output = predict_fn(model, tensor)
    if is_ragged:
        assert all(output.numpy() == np.array([[15.0]] * 3))
    else:
        assert all(output.numpy() == np.array([[15.0]]))


def test_tensorflow_v2_setup_run_batch():
    model_class = NativeModel()
    tag = bentoml.tensorflow.save_model(MODEL_NAME, model_class)
    runner = bentoml.tensorflow.get(tag).to_runner()

    assert tag in [model.tag for model in runner.models]
    runner.init_local()
    assert runner.run(native_data) == np.array([[15.0]])


@pytest.mark.gpus
def test_tensorflow_v2_setup_on_gpu():
    model_class = NativeModel()
    tag = bentoml.tensorflow.save_model(MODEL_NAME, model_class)
    runner = bentoml.tensorflow.get(tag).to_runner(nvidia_gpu=1)
    runner.init_local()
    # assert runner.num_replica == len(tf.config.list_physical_devices("GPU"))
    assert runner.run(native_tensor) == np.array([[15.0]])


# def test_tensorflow_v2_multi_args():
#     model_class = MultiInputModel()
#     tag = bentoml.tensorflow.save_model(MODEL_NAME, model_class)
#     runner1 = bentoml.tensorflow.load_runner(
#         tag,
#         partial_kwargs=dict(factor=tf.constant(3.0, dtype=tf.float64)),
#     )
#     runner2 = bentoml.tensorflow.load_runner(
#         tag,
#         partial_kwargs=dict(factor=tf.constant(2.0, dtype=tf.float64)),
#     )

#     assert runner1.run_batch(native_data, native_data) == np.array([[60.0]])
#     assert runner2.run_batch(native_data, native_data) == np.array([[45.0]])


@pytest.mark.parametrize("batch_dim", [0, 1])
def test_tensorflow_container(batch_dim: int):

    arr1 = np.ones((3, 3))
    if batch_dim == 0:
        arr2 = np.arange(6).reshape(2, 3)
    else:
        arr2 = np.arange(6).reshape(3, 2)

    tensor1: "tf_ext.TensorLike" = tf.convert_to_tensor(arr1, dtype=tf.float32)
    tensor2: "tf_ext.TensorLike" = tf.convert_to_tensor(arr2, dtype=tf.float32)

    batches = [tensor1, tensor2]
    batch, indices = TensorflowTensorContainer.batches_to_batch(
        batches, batch_dim=batch_dim
    )
    expected_batch: tf.Tensor = tf.concat(batches, axis=batch_dim)
    assert _tensor_equal(batch, expected_batch)

    restored_tensor1, restored_tensor2 = TensorflowTensorContainer.batch_to_batches(
        batch, indices, batch_dim=batch_dim
    )
    assert _tensor_equal(restored_tensor1, tensor1)
    assert _tensor_equal(restored_tensor2, tensor2)

    from_payload = TensorflowTensorContainer.from_payload(
        TensorflowTensorContainer.to_payload(tensor1)
    )
    assert _tensor_equal(from_payload, tensor1)

    restored_batch, restored_indices = TensorflowTensorContainer.from_batch_payloads(
        TensorflowTensorContainer.batch_to_payloads(
            batch, indices, batch_dim=batch_dim
        ),
        batch_dim=batch_dim,
    )
    assert restored_indices == indices
    assert _tensor_equal(restored_batch, batch)
