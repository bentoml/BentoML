from __future__ import annotations

import typing as t
import logging
import functools
from types import ModuleType
from pickle import UnpicklingError
from typing import TYPE_CHECKING

import msgpack.exceptions

import bentoml

from ..types import LazyType
from ..utils import LazyLoader
from ..utils.pkg import get_pkg_version
from .common.jax import jax
from .common.jax import jnp
from .common.jax import JaxArrayContainer
from ...exceptions import NotFound
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..models.model import ModelContext
from ..models.model import PartialKwargsModelOptions as ModelOptions
from ..runner.utils import Params

if TYPE_CHECKING:
    from flax import struct
    from jax.lib import xla_bridge
    from flax.core import FrozenDict
    from jax._src.lib.xla_bridge import XlaBackend

    from .. import external_typing as ext
    from ..tag import Tag
    from ...types import ModelSignature
    from ..models.model import ModelSignaturesType
else:
    xla_bridge = LazyLoader("xla_bridge", globals(), "jax.lib.xla_bridge")

try:
    from flax import linen as nn
    from flax import serialization
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "flax is required in order to use with 'bentoml.flax'. See https://flax.readthedocs.io/en/latest/index.html#installation for instructions."
    )

# NOTE: tensorflow is required since jax depends on XLA, which is a part of Tensorflow.
try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'tensorflow' is required in order to use module 'bentoml.flax', install tensorflow with 'pip install tensorflow'. For more information, refer to https://www.tensorflow.org/install"
    )


MODULE_NAME = "bentoml.flax"
MODEL_FILENAME = "saved_model.msgpack"
API_VERSION = "v1"

logger = logging.getLogger(__name__)


__all__ = ["load_model", "save_model", "get_runnable", "get", "JaxArrayContainer"]


def get(tag_like: str | Tag) -> bentoml.Model:
    """
    Get the BentoML model with the given tag.

    Args:
        tag_like: The tag of the model to retrieve from the model store.

    Returns:
        :obj:`~bentoml.Model`: A BentoML :obj:`~bentoml.Model` with the matching tag.

    Example:

    .. code-block:: python

       import bentoml

       model = bentoml.flax.get("mnist:latest")
    """
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, failed to load with {MODULE_NAME}."
        )
    return model


def load_model(
    bento_model: str | Tag | bentoml.Model,
    init: bool = True,
    device: str | XlaBackend = "cpu",
) -> tuple[nn.Module, dict[str, t.Any]]:
    """
    Load the ``flax.linen.Module`` model instance with the given tag from the local BentoML model store.

    Args:
        bento_model: Either the tag of the model to get from the store, or a BentoML `~bentoml.Model` instance to load the model from.
        init: Whether to initialize the state dict of given ``flax.linen.Module``.
              By default, the weights and values will be put to ``jnp.ndarray``. If ``init`` is set to ``False``,
              The state_dict will only be put to given accelerator device instead.
        device: The device to put the state dict to. By default, it will be put to ``cpu``. This is
                only used when ``init`` is set to ``False``.

    Returns:
        A tuple of ``flax.linen.Module`` as well as its ``state_dict`` from the model store.

    Example:

    .. code-block:: python

       import bentoml
       import jax

       net, state_dict = bentoml.flax.load_model("mnist:latest")
       predict_fn = jax.jit(lambda s: net.apply({"params": state_dict["params"]}, x))
       results = predict_fn(jnp.ones((1, 28, 28, 1)))
    """
    # NOTE: we need to hide all GPU from TensorFlow, otherwise it will try to allocate
    # memory on the GPU and make it unavailable for JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    if not isinstance(bento_model, bentoml.Model):
        bento_model = get(bento_model)
    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, failed loading with {MODULE_NAME}."
        )
    if "_module" not in bento_model.custom_objects:
        raise BentoMLException(
            f"Model {bento_model.tag} was either corrupt or not saved with 'bentoml.flax.save_model()'."
        )
    module: nn.Module = bento_model.custom_objects["_module"]

    serialized = bento_model.path_of(MODEL_FILENAME)
    try:
        with open(serialized, "rb") as f:
            state_dict: dict[str, t.Any] = serialization.from_bytes(module, f.read())
    except (UnpicklingError, msgpack.exceptions.ExtraData, UnicodeDecodeError) as err:
        raise BentoMLException(
            f"Unable to covert model {bento_model.tag}'s state_dict: {err}"
        ) from None

    # ensure that all arrays are restored as jnp.ndarray
    # NOTE: This is to prevent a bug this will be fixed in Flax >= v0.3.4:
    # https://github.com/google/flax/issues/1261
    if init:
        state_dict = jax.tree_util.tree_map(jnp.array, state_dict)
    else:
        # keep the params on given device if we don't want to initialize
        state_dict = jax.tree_util.tree_map(
            lambda s: jax.device_put(s, jax.devices(device)[0]), state_dict
        )
    return module, state_dict


def save_model(
    name: str,
    module: nn.Module,
    state: dict[str, t.Any] | FrozenDict[str, t.Any] | struct.PyTreeNode,
    *,
    signatures: ModelSignaturesType | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save a ``flax.linen.Module`` model instance to the BentoML model store.

    Args:
        name: The name to give to the model in the BentoML store. This must be a valid
              :obj:`~bentoml.Tag` name.
        module: ``flax.linen.Module`` to be saved.
        signatures: Signatures of predict methods to be used. If not provided, the signatures default to
                    ``predict``. See :obj:`~bentoml.types.ModelSignature` for more details.
        labels: A default set of management labels to be associated with the model. An example is ``{"training-set": "data-1"}``.
        custom_objects: Custom objects to be saved with the model. An example is ``{"my-normalizer": normalizer}``.
                        Custom objects are currently serialized with cloudpickle, but this implementation is subject to change.
        external_modules: user-defined additional python modules to be saved alongside the model or custom objects,
                          e.g. a tokenizer module, preprocessor module, model configuration module.
        metadata: Metadata to be associated with the model. An example is ``{"bias": 4}``.
                  Metadata is intended for display in a model management UI and therefore must be a
                  default Python type, such as :obj:`str` or :obj:`int`.

    Returns:
        :obj:`~bentoml.Tag`: A tag that can be used to access the saved model from the BentoML model store.

    Example:

    .. code-block:: python

       import jax

       rng, init_rng = jax.random.split(rng)
       state = create_train_state(init_rng, config)

       for epoch in range(1, config.num_epochs + 1):
           rng, input_rng = jax.random.split(rng)
           state, train_loss, train_accuracy = train_epoch(
               state, train_ds, config.batch_size, input_rng
           )
           _, test_loss, test_accuracy = apply_model(
               state, test_ds["image"], test_ds["label"]
           )

           logger.info(
               "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f",
               epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100
           )

       # `Save` the model with BentoML
       tag = bentoml.flax.save_model("mnist", CNN(), state)
    """
    if not isinstance(module, nn.Module):
        raise BentoMLException(
            f"'bentoml.flax.save_model()' only support saving 'flax.linen.Module' object. Got {module.__class__.__name__} instead."
        )

    context = ModelContext(
        framework_name="flax",
        framework_versions={
            "flax": get_pkg_version("flax"),
            "jax": get_pkg_version("jax"),
            "jaxlib": get_pkg_version("jaxlib"),
        },
    )

    if signatures is None:
        signatures = {"__call__": {"batchable": False}}
        logger.info(
            'Using the default model signature for Flax (%s) for model "%s".',
            signatures,
            name,
        )
    custom_objects = {} if custom_objects is None else custom_objects
    custom_objects["_module"] = module

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        options=ModelOptions(),
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
        context=context,
    ) as bento_model:
        with open(bento_model.path_of(MODEL_FILENAME), "wb") as f:
            f.write(serialization.to_bytes(state))
        return bento_model


def get_runnable(bento_model: bentoml.Model) -> t.Type[bentoml.Runnable]:
    """Private API: use :obj:`~bentoml.Model.to_runnable` instead."""
    partial_kwargs: dict[str, t.Any] = bento_model.info.options.partial_kwargs

    class FlaxRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("tpu", "nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()
            self.device = xla_bridge.get_backend().platform

            self.model, self.state_dict = load_model(bento_model, device=self.device)
            self.params = self.state_dict["params"]
            self.methods_cache: t.Dict[str, t.Callable[..., t.Any]] = {}

    def gen_run_method(self: FlaxRunnable, method_name: str):
        method = getattr(self.model, method_name)
        method_partial_kwargs = partial_kwargs.get(method_name)
        if method_partial_kwargs is not None:
            method = functools.partial(method, **method_partial_kwargs)

        def mapping(item: jnp.ndarray | ext.NpNDArray | ext.PdDataFrame) -> jnp.ndarray:
            if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(item):
                return jnp.asarray(item)
            if LazyType["ext.PdDataFrame"]("pandas.DataFrame").isinstance(item):
                # NOTE: only to_numpy() are doing copying in memory here.
                return jnp.asarray(item.to_numpy())
            return item

        def run_method(
            self: FlaxRunnable, *args: jnp.ndarray | ext.NpNDArray | ext.PdDataFrame
        ):
            params = Params[jnp.ndarray](*args).map(mapping)

            arg = params.args[0] if len(params.args) == 1 else params.args
            # NOTE: can we jit this?
            # No?, as we should not interfere with JAX tracing in multiple threads
            # https://jax.readthedocs.io/en/latest/concurrency.html?highlight=concurrency
            return self.model.apply({"params": self.params}, arg, method=method)

        return run_method

    def add_runnable_method(method_name: str, options: ModelSignature):
        def run_method(self: FlaxRunnable, *args: jnp.ndarray):
            fn = self.methods_cache.get(method_name)
            if not fn:
                fn = gen_run_method(self, method_name)
                self.methods_cache[method_name] = fn
            return fn(self, *args)

        FlaxRunnable.add_method(
            run_method,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return FlaxRunnable
