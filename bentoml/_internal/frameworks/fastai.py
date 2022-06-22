from __future__ import annotations

import types
import typing as t
import logging
import contextlib
from typing import TYPE_CHECKING

import cloudpickle

import bentoml

from ..types import LazyType
from ..utils.pkg import get_pkg_version
from ...exceptions import NotFound
from ...exceptions import InvalidArgument
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..models.model import ModelContext
from .common.pytorch import PyTorchTensorContainer
from ..runner.container import DataContainerRegistry

MODULE_NAME = "bentoml.fastai"
MODEL_FILENAME = "saved_model.pkl"
API_VERSION = "v1"

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from fastai.callback.core import Callback

    from .. import external_typing as ext
    from ..tag import Tag
    from ...types import ModelSignature
    from ..models.model import ModelSignaturesType

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "fastai requires `torch` as a dependency. Please follow PyTorch instruction at https://pytorch.org/ in order to use `fastai`."
    )
try:
    from fastai import __version__ as FASTAI_VERSION  # type: ignore
    from fastai.learner import Learner
    from fastai.learner import load_learner  # type: ignore
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "fastai is required in order to use module `bentoml.fastai`, install fastai with `pip install fastai`. For more information, refers to https://docs.fast.ai/#Installing."
    )

try:
    import fastai.basics  # type: ignore # noqa
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "BentoML only supports fastai v2 onwards. Please uninstall fastai v1 and try again. Please visit https://docs.fast.ai/#Installing for more information."
    )


__all__ = ["load_model", "save_model", "get_runnable", "get"]


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


def load_model(
    bento_model: str | Tag | bentoml.Model,
    cpu: bool = True,
) -> Learner:
    """
    Load the ``fastai.learner.Learner`` model instance with the given tag from the local BentoML model store.

    Args:
        bento_model: Either the tag of the model to get from the store, or a BentoML `~bentoml.Model` instance to load the model from.

        cpu: Whether to load the model to CPU. if true, the ``nn.Module`` is mapped to ``cpu`` via :code:`map_location` in ``torch.load``.
             The loaded dataloader will also be converted to ``cpu``.

             .. admonition:: About :code:`cpu=True`

                If the model uses ``mixed_precision``, then the loaded model will also be converted to FP32.
                Learn more about `mixed precision <https://docs.fast.ai/callback.fp16.html>`_.

    Returns:
        :code:`fastai.learner.Learner`:
            The :code:`fastai.learner.Learner` model instance loaded from the model store or BentoML :obj:`~bentoml.Model`.

    Example:

    .. code-block:: python

        import bentoml

        model = bentoml.fastai.load_model("fai_learner")
        results = model.predict({"input": "some input"})
    """  # noqa
    if not isinstance(bento_model, bentoml.Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, failed loading with {MODULE_NAME}."
        )

    return t.cast(Learner, load_learner(bento_model.path_of(MODEL_FILENAME), cpu=cpu))  # type: ignore (bad torch type)


def save_model(
    name: str,
    learner: Learner,
    *,
    pickle_module: types.ModuleType | None = None,
    signatures: ModelSignaturesType | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save a :code:`fastai.learner.Learner` model instance to the BentoML model store.

    Args:
        name: The name to give to the model in the BentoML store. This must be a valid
                    :obj:`~bentoml.Tag` name.
        learner: :obj:`~fastai.learner.Learner` to be saved.
        pickle_module: The pickle module to use for exporting the model.

                            .. admonition:: About pickling :obj:`~fastai.learner.Learner`

                            If the :func:`save_model` failed while saving a given learner, then
                            your learner contains :obj:`~fastai.callback.core.Callback` that is not pickable.
                            This is due to all FastAI callbacks are stateful, which makes some of them not pickable.
                            To fix this, Use :func:`Learner.remove_cbs` to remove list of training callback that fails.

        signatures: Signatures of predict methods to be used. If not provided, the signatures default to
                        ``predict``. See :obj:`~bentoml.types.ModelSignature` for more details.
        labels: A default set of management labels to be associated with the model. An example is ``{"training-set": "data-1"}``.
        custom_objects: Custom objects to be saved with the model. An example is ``{"my-normalizer": normalizer}``.
                            Custom objects are currently serialized with cloudpickle, but this implementation is subject to change.
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
    if isinstance(learner, nn.Module):
        raise BentoMLException(
            "'bentoml.fastai.save_model()' does not support saving pytorch 'Module's directly. You should create a new 'Learner' object from the model, or use 'bentoml.pytorch.save_model()' to save your PyTorch model instead."
        )
    if not isinstance(learner, Learner):
        raise BentoMLException(
            f"'bentoml.fastai.save_model()' only support saving fastai 'Learner' object. Got {type(learner)} instead."
        )

    context = ModelContext(
        framework_name="fastai",
        framework_versions={
            "fastai": FASTAI_VERSION,
            "fastcore": get_pkg_version("fastcore"),
            "torch": get_pkg_version("torch"),
        },
    )

    if signatures is None:
        signatures = {"predict": {"batchable": False}}
        logger.info(
            f"Using the default model signature ({signatures}) for model {name}."
        )

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        options=None,
        custom_objects=custom_objects,
        metadata=metadata,
        context=context,
    ) as bento_model:
        from fastai.callback.schedule import ParamScheduler

        if pickle_module is None:
            pickle_module = cloudpickle

        # NOTE: ParamScheduler is not pickable, so we need to remove it from the model.
        # It is also a hyperparameter callback, hence we don't need it for serving.
        cbs: list[Callback] = list(filter(lambda x: isinstance(x, ParamScheduler), learner.cbs))  # type: ignore (bad fastai type)
        learner.remove_cbs(cbs)
        learner.export(bento_model.path_of(MODEL_FILENAME), pickle_module=pickle_module)
        learner.add_cbs(cbs)
        return bento_model


def get_runnable(bento_model: bentoml.Model) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """
    logger.warning(
        "Runners created from FastAIRunnable will not be optimized for performance. If performance is critical to your usecase, please access the PyTorch model directly via 'learn.model' and use 'bentoml.pytorch.get_runnable()' instead."
    )

    class FastAIRunnable(bentoml.Runnable):
        SUPPORT_NVIDIA_GPU = False
        SUPPORT_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()

            if torch.cuda.is_available():
                logger.debug(
                    "CUDA is available, but BentoML does not support running fastai models on GPU."
                )
            self.learner = load_model(bento_model)
            self.learner.model.train(False)  # to turn off dropout and batchnorm
            self._no_grad_context = contextlib.ExitStack()
            if hasattr(torch, "inference_mode"):  # pytorch>=1.9
                self._no_grad_context.enter_context(torch.inference_mode())
            else:
                self._no_grad_context.enter_context(torch.no_grad())

            # TODO: support GPU and cpu=False
            self.device = "cpu"

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in bento_model.info.signatures:
                try:
                    self.predict_fns[method_name] = getattr(self.learner, method_name)
                except AttributeError:
                    raise InvalidArgument(
                        f"No method with name {method_name} found for Learner of type {self.learner.__class__}"
                    )

        def __del__(self):
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


# NOTE: we are currently register DataContainer
#   for each of the data container type on the module level code.
#
# By simply have the line below, we are already registering
#   PyTorchTensorContainer due to Python's side-effect of imports.
#
# However, we will still register PyTorchTensorContainer at the end
#   of this file for consistency. Since the map of DataContainerRegisty's
#   single type and batch type is a dictionary of LazyType and the container
#   itself, it would just replce the existing value.
#
# This operation is O(1), hence no need to worry about performance.
DataContainerRegistry.register_container(
    LazyType("torch", "Tensor"),
    LazyType("torch", "Tensor"),
    PyTorchTensorContainer,
)
