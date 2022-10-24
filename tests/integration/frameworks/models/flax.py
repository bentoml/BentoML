from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from jax.nn import initializers
from flax.core import Scope

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
    def __call__(  # pytlint: disable=arguments-differ
        self, x1: jnp.ndarray, x2: jnp.ndarray, features: int = 3
    ) -> jnp.ndarray:
        x: jnp.ndarray = jnp.concatenate([x1, x2], axis=-1)
        x = Dense(features)(x)
        x = Dense(features)(x)
        return x


ones_array = jnp.ones((10,))


def make_mlp_model():
    rngkey = random.PRNGKey(0)
    scope = Scope({}, {"params": rngkey}, mutable=["params"])
    net = MLP(parent=scope)
    print(net(ones_array))
    return net


mlp = FrameworkTestModel(
    name="mlp",
    model=make_mlp_model(),
    save_kwargs={
        "state": make_mlp_model()(ones_array),
        "signatures": {"__call__": {"batchable": True, "batch_dim": 0}},
    },
    configurations=[
        Config(
            load_kwargs={},
            test_inputs={
                "__call__": [
                    Input(
                        input_args=[[ones_array]],
                        expected=lambda out: np.testing.assert_allclose(
                            out, jnp.array([2.0])
                        ),
                    )
                ]
            },
        )
    ],
)

models = [mlp]
