from __future__ import annotations

import typing as t
import logging
import contextlib
from types import ModuleType
from typing import TYPE_CHECKING

import bentoml

from ..utils.pkg import get_pkg_version
from ...exceptions import NotFound
from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..models.model import ModelContext

# register PyTorchTensorContainer as import side effect.
from .common.pytorch import PyTorchTensorContainer

MODULE_NAME = "bentoml.fastai"
MODEL_FILENAME = "saved_model.pkl"
API_VERSION = "v1"

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .. import external_typing as ext
    from ..tag import Tag
    from ...types import ModelSignature
    from ..models.model import ModelSignaturesType

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "fastai requires 'torch' as a dependency. Please follow PyTorch instruction at https://pytorch.org/get-started/locally/ in order to use 'fastai'."
    )

try:
    import fastai.learner as learner
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'fastai' is required in order to use module 'bentoml.fastai'. Install fastai with 'pip install fastai'. For more information, refer to https://docs.fast.ai/#Installing."
    )

try:
    import fastai.basics as _  # noqa
except ImportError:  # pragma: no cover
    raise MissingDependencyException("BentoML only supports fastai v2 onwards.")


__all__ = ["load_model", "save_model", "get_runnable", "get", "PyTorchTensorContainer"]


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
       # target model must be from the BentoML model store
       model = bentoml.fastai.get("fai_learner")
    """
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, failed to load with {MODULE_NAME}."
        )
    return model


def load_model(bento_model: str | Tag | bentoml.Model) -> learner.Learner:
    """
    Load the ``fastai.learner.Learner`` model instance with the given tag from the local BentoML model store.

    If the model uses ``mixed_precision``, then the loaded model will also be converted to FP32. Learn more about `mixed precision <https://docs.fast.ai/callback.fp16.html>`_.

    Args:
        bento_model: Either the tag of the model to get from the store, or a BentoML `~bentoml.Model` instance to load the model from.

    Returns:
        :code:`fastai.learner.Learner`:
            The :code:`fastai.learner.Learner` model instance loaded from the model store or BentoML :obj:`~bentoml.Model`.

    Example:

    .. code-block:: python

       import bentoml

       model = bentoml.fastai.load_model("fai_learner")
       results = model.predict("some input")
    """  # noqa

    if not isinstance(bento_model, bentoml.Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, failed loading with {MODULE_NAME}."
        )

    pickle_file: str = bento_model.path_of(MODEL_FILENAME)
    with open(pickle_file, "rb") as f:
        return t.cast(learner.Learner, learner.load_learner(f, cpu=True))


def save_model(
    name: Tag | str,
    learner_: learner.Learner,
    *,
    signatures: ModelSignaturesType | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save a :code:`fastai.learner.Learner` model instance to the BentoML model store.

    If the :func:`save_model` method failed while saving a given learner,
    your learner may contain a :obj:`~fastai.callback.core.Callback` that is not picklable.
    All FastAI callbacks are stateful, which makes some of them not picklable.
    Use :func:`Learner.remove_cbs` to remove unpicklable callbacks.

    Args:
        name: The name to give to the model in the BentoML store. This must be a valid
              :obj:`~bentoml.Tag` name.
        learner: :obj:`~fastai.learner.Learner` to be saved.
        signatures: Signatures of predict methods to be used. If not provided, the signatures default to
                    ``predict``. See :obj:`~bentoml.types.ModelSignature` for more details.
        labels: A default set of management labels to be associated with the model. An example is ``{"training-set": "data-1"}``.
        custom_objects: Custom objects to be saved with the model. An example is ``{"my-normalizer": normalizer}``.
                        Custom objects are currently serialized with cloudpickle, but this implementation is subject to change.
        external_modules (:code:`List[ModuleType]`, `optional`, default to :code:`None`):
            user-defined additional python modules to be saved alongside the model or custom objects,
            e.g. a tokenizer module, preprocessor module, model configuration module
        metadata: Metadata to be associated with the model. An example is ``{"bias": 4}``.
                  Metadata is intended for display in a model management UI and therefore must be a
                  default Python type, such as :obj:`str` or :obj:`int`.

    Returns:
        :obj:`~bentoml.Tag`: A tag that can be used to access the saved model from the BentoML model store.

    Example:

    .. code-block:: python

       from fastai.metrics import accuracy
       from fastai.text.data import URLs
       from fastai.text.data import untar_data
       from fastai.text.data import TextDataLoaders
       from fastai.text.models import AWD_LSTM
       from fastai.text.learner import text_classifier_learner

       dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid="test")

       learner = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
       learner.fine_tune(4, 1e-2)

       # Test run the model
       learner.model.eval()
       learner.predict("I love that movie!")

       # `Save` the model with BentoML
       tag = bentoml.fastai.save_model("fai_learner", learner)
    """
    import cloudpickle

    if isinstance(learner_, nn.Module):
        raise BentoMLException(
            "'bentoml.fastai.save_model()' does not support saving pytorch 'Module's directly. You should create a new 'Learner' object from the model, or use 'bentoml.pytorch.save_model()' to save your PyTorch model instead."
        )
    if not isinstance(learner_, learner.Learner):
        raise BentoMLException(
            f"'bentoml.fastai.save_model()' only support saving fastai 'Learner' object. Got {learner.__class__.__name__} instead."
        )

    context = ModelContext(
        framework_name="fastai",
        framework_versions={
            "fastai": get_pkg_version("fastai"),
            "fastcore": get_pkg_version("fastcore"),
            "torch": get_pkg_version("torch"),
        },
    )

    if signatures is None:
        signatures = {"predict": {"batchable": False}}
        logger.info(
            'Using the default model signature for fastai (%s) for model "%s".',
            signatures,
            name,
        )
    batchable_enabled_signatures = [v for v in signatures if signatures[v]["batchable"]]
    if len(batchable_enabled_signatures) > 0:
        message = f"Batchable signatures are not supported for fastai models. The following signatures have batchable sets to 'True': {batchable_enabled_signatures}. Consider using PyTorch layer from the learner model. To learn more, visit https://docs.bentoml.org/en/latest/frameworks/fastai.html."
        raise BentoMLException(message)

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
        learner_.export(bento_model.path_of(MODEL_FILENAME), pickle_module=cloudpickle)
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
