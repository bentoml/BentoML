from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tensorflow as tf

from bentoml._internal.types import LazyType
from bentoml._internal.runner.container import AutoContainer
from bentoml._internal.runner.container import DataContainerRegistry

if TYPE_CHECKING:
    from bentoml._internal.external_typing import tensorflow as ext

    P = t.ParamSpec("P")


def requires_disable_eager_execution(
    function: t.Callable[P, None]
) -> t.Callable[P, None]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        if tf.executing_eagerly():
            pytest.skip(
                "This test requires disable eager execution with 'tf.compat.v1.disable_eager_execution()'"
            )
        return function(*args, **kwargs)

    return wrapper


def assert_tensor_equal(t1: ext.TensorLike, t2: ext.TensorLike) -> None:
    assert t1.shape == t2.shape
    assert tf.math.equal(t1, t2).numpy().all()


@pytest.mark.requires_eager_execution
@pytest.mark.parametrize("batch_axis", [0, 1])
def test_tensorflow_container(batch_axis: int):

    from bentoml._internal.frameworks.tensorflow_v2 import TensorflowTensorContainer

    one_batch: ext.TensorLike = tf.reshape(tf.convert_to_tensor(np.arange(6)), (2, 3))
    batch_list: list[ext.TensorLike] = [one_batch, one_batch + 1]
    merged_batch = tf.concat(batch_list, batch_axis)

    batches, indices = TensorflowTensorContainer.batches_to_batch(
        batch_list,
        batch_dim=batch_axis,
    )
    assert batches.shape == merged_batch.shape
    assert_tensor_equal(batches, merged_batch)
    assert_tensor_equal(
        TensorflowTensorContainer.batch_to_batches(
            merged_batch,
            indices=indices,
            batch_dim=batch_axis,
        )[0],
        one_batch,
    )
    assert_tensor_equal(
        TensorflowTensorContainer.from_payload(
            TensorflowTensorContainer.to_payload(one_batch)
        ),
        one_batch,
    )

    assert_tensor_equal(
        AutoContainer.from_payload(AutoContainer.to_payload(one_batch, batch_dim=0)),
        one_batch,
    )


@requires_disable_eager_execution
def test_register_container():

    assert not tf.executing_eagerly()

    from bentoml._internal.frameworks.tensorflow_v2 import (  # type: ignore # noqa
        TensorflowTensorContainer,
    )

    assert (
        LazyType("tensorflow.python.framework.ops", "Tensor")
        not in DataContainerRegistry.CONTAINER_BATCH_TYPE_MAP
    )

    assert (
        LazyType("tensorflow.python.framework.ops", "Tensor")
        not in DataContainerRegistry.CONTAINER_SINGLE_TYPE_MAP
    )
