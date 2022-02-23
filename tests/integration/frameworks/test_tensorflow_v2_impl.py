from typing import Any
from typing import List
from typing import Union
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_hub as hub

import bentoml
from tests.utils.helpers import assert_have_file_extension
from tests.utils.frameworks.tensorflow_utils import NativeModel
from tests.utils.frameworks.tensorflow_utils import MultiInputModel
from tests.utils.frameworks.tensorflow_utils import NativeRaggedModel
from tests.utils.frameworks.tensorflow_utils import KerasSequentialModel

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
    tag = bentoml.tensorflow.save(MODEL_NAME, mcls)
    _model = bentoml.models.get(tag)
    assert_have_file_extension(_model.path, ".pb")
    model = bentoml.tensorflow.load(MODEL_NAME)
    output = predict_fn(model, tensor)
    if is_ragged:
        assert all(output.numpy() == np.array([[15.0]] * 3))
    else:
        assert all(output.numpy() == np.array([[15.0]]))


def test_tensorflow_v2_setup_run_batch():
    model_class = NativeModel()
    tag = bentoml.tensorflow.save(MODEL_NAME, model_class)
    runner = bentoml.tensorflow.load_runner(tag)

    assert tag in runner.required_models
    assert runner.num_replica == 1
    assert runner.run_batch(native_data) == np.array([[15.0]])


@pytest.mark.gpus
def test_tensorflow_v2_setup_on_gpu():
    model_class = NativeModel()
    tag = bentoml.tensorflow.save(MODEL_NAME, model_class)
    runner = bentoml.tensorflow.load_runner(tag)

    assert runner.num_replica == len(tf.config.list_physical_devices("GPU"))
    assert runner.run_batch(native_tensor) == np.array([[15.0]])


def test_tensorflow_v2_multi_args():
    model_class = MultiInputModel()
    tag = bentoml.tensorflow.save(MODEL_NAME, model_class)
    runner1 = bentoml.tensorflow.load_runner(
        tag,
        partial_kwargs=dict(factor=tf.constant(3.0, dtype=tf.float64)),
    )
    runner2 = bentoml.tensorflow.load_runner(
        tag,
        partial_kwargs=dict(factor=tf.constant(2.0, dtype=tf.float64)),
    )

    assert runner1.run_batch(native_data, native_data) == np.array([[60.0]])
    assert runner2.run_batch(native_data, native_data) == np.array([[45.0]])


def _plus_one_model_tf2():
    obj = tf.train.Checkpoint()

    @tf.function(input_signature=[tf.TensorSpec(None, dtype=tf.float32)])
    def plus_one(x):
        return x + 1

    obj.__call__ = plus_one
    return obj


def _plus_one_model_tf1():
    def plus_one():
        x = tf.compat.v1.placeholder(dtype=tf.float32, name="x")
        y = x + 1
        hub.add_signature(inputs=x, outputs=y)

    spec = hub.create_module_spec(plus_one)
    with tf.compat.v1.get_default_graph().as_default():
        module = hub.Module(spec, trainable=True)
        return module


@pytest.mark.parametrize(
    "identifier, name, tags, is_module_v1, wrapped",
    [
        (_plus_one_model_tf1(), "module_hub_tf1", [], True, False),
        (_plus_one_model_tf2(), "saved_model_tf2", ["serve"], False, False),
        (
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            None,
            None,
            False,
            True,
        ),
    ],
)
def test_import_from_tfhub(
    identifier: Union[Callable[[], Union["hub.Module", "hub.KerasLayer"]], str],
    name: Optional[str],
    tags: Optional[List[Any]],
    is_module_v1: bool,
    wrapped: bool,
):
    if isinstance(identifier, str):
        import tensorflow_text as text  # noqa # pylint: disable

    tag = bentoml.tensorflow.import_from_tfhub(identifier, name)
    model = bentoml.models.get(tag)
    assert model.info.context["import_from_tfhub"]
    module = bentoml.tensorflow.load(tag, tags=tags, load_as_hub_module=wrapped)
    assert module._is_hub_module_v1 == is_module_v1  # pylint: disable
