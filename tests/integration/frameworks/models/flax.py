from __future__ import annotations

import typing as t

import chex
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from jax.nn import initializers

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.flax

backward_compatible = False


class Dense(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param(
            "kernel", initializers.lecun_normal(), (x.shape[-1], self.features)
        )
        y: jnp.ndarray = jnp.dot(x, kernel)
        return y


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = Dense(3)(x)
        x = Dense(3)(x)
        return x


class MultiInputPerceptron(nn.Module):
    @nn.compact
    def __call__(
        self, x1: jnp.ndarray, x2: jnp.ndarray, features: int = 3
    ) -> jnp.ndarray:
        x: jnp.ndarray = jnp.concatenate([x1, x2], axis=-1)
        x = Dense(features)(x)
        x = Dense(features)(x)
        return x


ones_array = jnp.ones((10,))
prngkey = random.PRNGKey(42)


def init_mlp_state():
    net = MLP()
    params = net.init({"params": prngkey}, ones_array)
    return params


def assert_equal_shape(
    model: nn.Module, state_dict: dict[str, t.Any], arr: jnp.ndarray
):
    def check(out: jnp.ndarray) -> bool:
        logit = model.apply({"params": state_dict["params"]}, arr)
        chex.assert_equal_shape([logit, out])
        assert (logit == out).all()
        return True

    return check


mlp = FrameworkTestModel(
    name="mlp",
    model=MLP(),
    save_kwargs={
        "state": init_mlp_state(),
        "signatures": {"__call__": {"batchable": True, "batch_dim": 0}},
    },
    configurations=[
        Config(
            load_kwargs={},
            test_inputs={
                "__call__": [
                    Input(
                        input_args=[ones_array],
                        expected=assert_equal_shape(
                            MLP(), init_mlp_state(), ones_array
                        ),
                    )
                ]
            },
        )
    ],
)

models = [mlp]
