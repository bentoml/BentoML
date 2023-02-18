# modified from https://github.com/google/flax/blob/main/examples/mnist/README.md
from __future__ import annotations

import os
import typing as t
import argparse
from typing import TYPE_CHECKING

import jax
import attrs
import numpy as np
import optax
import cattrs
import jax.numpy as jnp
import tensorflow_datasets as tfds
from flax import linen as nn
from flax import serialization
from flax.metrics import tensorboard
from flax.training import train_state

import bentoml
from bentoml._internal.utils.pkg import pkg_version_info

if TYPE_CHECKING:
    import tensorflow as tf
    from flax import core
    from jax._src.random import KeyArray
    from tensorflow_datasets.core.utils.type_utils import Tree

    from bentoml._internal import external_typing as ext

    NumpyElem = ext.NpNDArray | tf.RaggedTensor


@attrs.define
class ConfigDict:
    learning_rate: float = 0.1
    batch_size: int = 128
    num_epochs: int = 10
    momentum: float = 0.9
    enable_tensorboard: bool = True
    hermetic: bool = True

    def to_dict(self) -> dict[str, t.Any]:
        return cattrs.unstructure(self)

    def with_options(self, **kwargs: float | int | bool) -> ConfigDict:
        return attrs.evolve(self, **kwargs)


_DefaultConfig = ConfigDict()


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # pylint: disable=W0221
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


@jax.jit
def apply_model(
    state: train_state.TrainState, image: jnp.DeviceArray, labels: jnp.DeviceArray
) -> tuple[t.Callable[..., core.FrozenDict[str, t.Any]], jnp.ndarray, jnp.ndarray]:
    """Compute gradients, loss, and accuracy for a single batch."""

    def loss_fn(params: core.FrozenDict[str, t.Any]) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits: jnp.ndarray = state.apply_fn({"params": params}, image)
        one_hot = jax.nn.one_hot(labels, 10)  # 0 to 9
        loss: jnp.ndarray = jnp.mean(
            optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        )
        return loss, logits

    grad = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grad = grad(state.params)
    accuracy: jnp.ndarray = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grad, loss, accuracy


@jax.jit
def update_model(state: train_state.TrainState, grads: core.FrozenDict[str, t.Any]):
    return state.apply_gradients(grads=grads)


def train_epoch(
    state: train_state.TrainState,
    train_ds: Tree[NumpyElem],
    batch_size: int,
    rng: KeyArray,
):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds["image"][perm, ...]
        batch_labels = train_ds["label"][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def get_datasets() -> tuple[Tree[NumpyElem], Tree[NumpyElem]]:
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
    train_ds["image"] = jnp.float32(train_ds["image"]) / 255.0
    test_ds["image"] = jnp.float32(test_ds["image"]) / 255.0
    return train_ds, test_ds


def create_train_state(rng: KeyArray, config: ConfigDict) -> train_state.TrainState:
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def train_and_evaluate(
    config: ConfigDict = _DefaultConfig, workdir: str = "."
) -> train_state.TrainState:
    """Execute model training and evaluation loop.
    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(0)
    summary_writer: tensorboard.SummaryWriter | None = None

    if config.enable_tensorboard:
        summary_writer = tensorboard.SummaryWriter(workdir)
        summary_writer.hparams(config.to_dict())

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        if not config.hermetic:
            rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state, train_ds, config.batch_size, input_rng
        )
        _, test_loss, test_accuracy = apply_model(
            state, test_ds["image"], test_ds["label"]
        )

        print(
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
            % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
        )

        if config.enable_tensorboard:
            assert summary_writer is not None
            summary_writer.scalar("train_loss", train_loss, epoch)
            summary_writer.scalar("train_accuracy", train_accuracy, epoch)
            summary_writer.scalar("test_loss", test_loss, epoch)
            summary_writer.scalar("test_accuracy", test_accuracy, epoch)

    if config.enable_tensorboard:
        assert summary_writer is not None
        summary_writer.flush()

    return state


def load_and_predict(path: str, idx: int = 0):
    """
    Load the saved msgpack model and make predictions.
    We will run prediction on test MNIST dataset
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at {path}")
    with open(path, "rb") as f:
        state_dict = serialization.from_bytes(CNN, f.read())
    cnn = CNN()
    _, test_ds = get_datasets()
    # ensure that all arrays are restored as jnp.ndarray
    # NOTE: This is to prevent a bug this will be fixed in Flax >= v0.3.4:
    # https://github.com/google/flax/issues/1261
    if pkg_version_info("flax") < (0, 3, 4):
        state_dict = jax.tree_util.tree_map(jnp.ndarray, state_dict)
    # jit it !
    logits = jax.jit(lambda x: cnn.apply({"params": state_dict["params"]}, x))(
        test_ds["image"]
    )
    return logits[idx].argmax()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.94)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--enable-tensorboard", action="store_true")
    parser.add_argument(
        "--hermetic",
        action="store_true",
        default=False,
        help="Whether to use random key",
    )
    args = parser.parse_args()

    training_state = train_and_evaluate(
        config=_DefaultConfig.with_options(
            learning_rate=args.lr,
            momentum=args.momentum,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            enable_tensorboard=args.enable_tensorboard,
            hermetic=args.hermetic,
        ),
    )

    model = bentoml.flax.save_model("mnist_flax", CNN(), training_state)
    print(f"Saved model: {model}")
