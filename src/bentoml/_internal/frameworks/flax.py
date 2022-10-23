from __future__ import annotations

import typing as t
import logging
import contextlib
from types import ModuleType
from typing import TYPE_CHECKING

import bentoml
import msgpack.exceptions

from ..utils.pkg import pkg_version_info
from .common.jax import jnp, jaxlib, jax
from .common.jax import JaxArrayContainer
from ...exceptions import NotFound
from ...exceptions import InvalidArgument, BentoMLException
from ...exceptions import MissingDependencyException
from ..models.model import ModelContext
from ..runner.utils import Params

MODULE_NAME = "bentoml.flax"
MODEL_FILENAME = "saved_model.msgpack"
API_VERSION = "v1"

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .. import external_typing as ext
    from ..tag import Tag
    from ...types import ModelSignature
    from ..models.model import ModelSignaturesType

try:
    import flax.version
    from flax import linen as nn
    from flax.core import FrozenDict, freeze
    from flax import serialization
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "flax is required in order to use with 'bentoml.flax'. See https://flax.readthedocs.io/en/latest/index.html#installation for instructions."
    )


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


def load_model(bento_model: str | Tag | bentoml.Model) -> nn.Module:
    """
    Load the ``flax.linen.Module`` model instance with the given tag from the local BentoML model store.

    Args:
        bento_model: Either the tag of the model to get from the store, or a BentoML `~bentoml.Model` instance to load the model from.

    Returns:
        ``flax.linen.Module``:
            The ``flax.linen.Module`` model instance loaded from the model store or BentoML :obj:`~bentoml.Model`.

    Example:

    .. code-block:: python

       import bentoml

       model = bentoml.flax.load_model("mnist:latest")
       results = model.predict("some input")
    """  # noqa

    if not isinstance(bento_model, bentoml.Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, failed loading with {MODULE_NAME}."
        )
    if "_flax_module" not in bento_model.custom_objects:
        raise BentoMLException(
            f"Model {bento_model.tag} was either corrupt or not saved with 'bentoml.flax.save_model()'."
        )
    module: nn.Module = bento_model.custom_objects["_flax_module"]

    serialized = bento_model.path_of(MODEL_FILENAME)
    with open(serialized, "rb") as f:
        state_dict = serialization.from_bytes(module, f.read())

    # ensure that all arrays are restored as jnp.ndarray
    # NOTE: This is to prevent a bug this will be fixed in Flax >= v0.3.4:
    # https://github.com/google/flax/issues/1261
    if pkg_version_info("flax") < (0, 3, 4):
        state_dict = jax.tree_util.tree_map(jnp.ndarray, state_dict)


def save_model(
    name: str,
    module: nn.Module,
    state: dict[str, t.Any] | FrozenDict[str, t.Any],
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
               "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f"
               % (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100)
           )

       # `Save` the model with BentoML
       tag = bentoml.fastai.save_model("fai_learner", learner)
    """
    if not isinstance(module, nn.Module):
        raise BentoMLException(
            f"'bentoml.flax.save_model()' only support saving 'flax.linen.Module' object. Got {module.__class__.__name__} instead."
        )
    if not isinstance(state, FrozenDict):
        state = freeze(state)

    context = ModelContext(
        framework_name="flax",
        framework_versions={
            "flax": flax.version.__version__,
            "jax": jax.__version__,
            "jaxlib": jaxlib.version.__version__,
        },
    )

    if signatures is None:
        signatures = {"predict": {"batchable": False}}
        logger.info(
            'Using the default model signature for flax (%s) for model "%s".',
            signatures,
            name,
        )
    custom_objects = {} if custom_objects is None else custom_objects
    custom_objects["_flax_module"] = module

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        options=None,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
        context=context,
    ) as bento_model:
        with open(bento_model.path_of(MODEL_FILENAME), "wb") as f:
            f.write(serialization.to_bytes(state))
        return bento_model


def get_runnable(bento_model: bentoml.Model) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """
    logger.warning(
        "Runners created from FastAIRunnable will not be optimized for performance. If performance is critical to your usecase, please access the PyTorch model directly via 'learn.model' and use 'bentoml.pytorch.get_runnable()' instead."
    )

    class FastAIRunnable(bentoml.Runnable):
        # fastai only supports GPU during training, not for inference.
        SUPPORTED_RESOURCES = ("cpu",)
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()

            if torch.cuda.is_available():
                logger.debug(
                    "CUDA is available, but BentoML does not support running fastai models on GPU."
                )
            self.learner = load_model(bento_model)
            self.learner.model.train(False)  # type: ignore (bad pytorch type) # to turn off dropout and batchnorm
            self._no_grad_context = contextlib.ExitStack()
            if hasattr(torch, "inference_mode"):  # pytorch>=1.9
                self._no_grad_context.enter_context(torch.inference_mode())
            else:
                self._no_grad_context.enter_context(torch.no_grad())

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in bento_model.info.signatures:
                try:
                    self.predict_fns[method_name] = getattr(self.learner, method_name)
                except AttributeError:
                    raise InvalidArgument(
                        f"No method with name {method_name} found for Learner of type {self.learner.__class__}"
                    )

        def __del__(self):
            if hasattr(self, "_no_grad_context"):
                self._no_grad_context.close()

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(
            self: FastAIRunnable,
            input_data: ext.NpNDArray | torch.Tensor | ext.PdSeries | ext.PdDataFrame,
        ) -> torch.Tensor:
            return self.predict_fns[method_name](input_data)

        FastAIRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return FastAIRunnable
