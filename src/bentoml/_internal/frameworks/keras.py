from __future__ import annotations

import functools
import logging
import typing as t
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING

import attr
import keras
from packaging import version

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException
from bentoml.exceptions import NotFound
from bentoml.legacy import Runnable
from bentoml.models import ModelContext

from ..models.model import ModelSignature
from ..models.model import PartialKwargsModelOptions
from ..runner.utils import Params
from ..types import LazyType
from ..utils.pkg import get_pkg_version
from .utils.tensorflow import get_tf_version

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from .. import external_typing as ext
    from ..external_typing import tensorflow as tf_ext
    from ..models.model import ModelSignatureDict

    KerasArgType = t.Union[t.List[t.Union[int, float]], ext.NpNDArray, "tf_ext.Tensor"]


MODULE_NAME = "bentoml.keras"
API_VERSION = "v1"


def _get_tf() -> t.Any:
    """Lazily import TensorFlow; return None if it is not installed."""
    try:
        import tensorflow as tf

        return tf
    except ImportError:
        return None


def _get_keras_backend() -> str:
    """Return the active Keras backend name.

    Keras 3 exposes ``keras.config.backend()``. Earlier versions only support the
    TensorFlow backend.
    """
    try:
        if version.parse(keras.__version__) >= version.parse("3.0.0"):
            return keras.config.backend()
    except Exception:
        pass
    return "tensorflow"


def _get_context(backend: str) -> ModelContext:
    """Build a ModelContext that records the backend and relevant package versions."""
    framework_versions: t.Dict[str, str] = {
        "keras": keras.__version__,
        "backend": backend,
    }
    if backend == "tensorflow":
        tf_version = get_tf_version()
        if tf_version:
            framework_versions["tensorflow"] = tf_version
    elif backend == "torch":
        framework_versions["torch"] = get_pkg_version("torch")
    elif backend == "jax":
        try:
            framework_versions["jax"] = get_pkg_version("jax")
        except Exception:
            pass
    return ModelContext(framework_name="keras", framework_versions=framework_versions)


@attr.define
class ModelOptions(PartialKwargsModelOptions):
    """Options for the Keras model."""

    include_optimizer: bool = False


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
       model = bentoml.keras.get("keras_resnet50")
    """
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def _get_default_device(backend: str) -> str:
    """Return a default device string appropriate for the active backend."""
    if backend == "tensorflow":
        tf = _get_tf()
        if tf is not None and len(tf.config.list_physical_devices("GPU")) > 0:
            # In Multi-GPU scenarios, the visible cuda devices will be set for each Runner worker
            # by the runner's Scheduling Strategy. So that the Runnable implementation only needs
            # to find the first GPU device visible to current process.
            return "/device:GPU:0"
        return "/device:CPU:0"
    if backend == "torch":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _normalize_device_name(backend: str, device_name: str) -> str:
    """Return a device string suitable for the active backend.

    TensorFlow-style device strings such as ``"/device:GPU:0"`` are accepted as
    the default and are converted to backend-specific strings when needed (e.g.
    ``"cuda"`` for PyTorch).
    """
    if backend == "torch":
        if device_name.startswith("/device:GPU"):
            return "cuda"
        if device_name.startswith("/device:CPU"):
            return "cpu"
    return device_name


@contextmanager
def _device_scope(backend: str, device_name: str):
    """Backend-specific device scope for model loading.

    TensorFlow supports a global device context. Other backends handle device
    placement inside the runnable by moving tensors explicitly.
    """
    if backend == "tensorflow":
        tf = _get_tf()
        if tf is None:
            yield
            return
        with tf.device(device_name):
            yield
    else:
        yield


def load_model(
    bento_model: str | Tag | bentoml.Model,
    device_name: str = "/device:CPU:0",
) -> "keras.Model":
    """
    Load a model from BentoML local modelstore with given name.

    Keras 3 models can only be loaded with the same backend that was active
    when the model was saved (``tensorflow``, ``torch``, ``jax``, ...). The
    current backend is read from ``keras.config.backend()``; if it differs from
    the saved backend, a ``BentoMLException`` is raised. TensorFlow is only
    required when loading a model saved with the TensorFlow backend.

    Args:
        bento_model (``str`` ``|`` :obj:`~bentoml.Tag` ``|`` :obj:`~bentoml.Model`):
            Either the tag of the model to get from the store, or a BentoML `~bentoml.Model`
            instance to load the model from.
        device_name (``str`` | ``None``):
            The device id to load the model on. For the TensorFlow backend the format should be
            compatible with `tf.device <https://www.tensorflow.org/api_docs/python/tf/device>`_.
            For the PyTorch backend strings such as ``"cpu"`` and ``"cuda"`` are accepted.

    Returns:
        :obj:`keras.Model`: an instance of users :obj:`keras.Model` from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        # load a model back into memory:
        loaded = bentoml.keras.load_model("keras_model")

    """  # noqa

    if not isinstance(bento_model, bentoml.Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )

    backend = _get_keras_backend()
    saved_backend = bento_model.info.context.framework_versions.get("backend")
    if saved_backend is not None and saved_backend != backend:
        raise BentoMLException(
            f"Keras model {bento_model.tag} was saved with backend '{saved_backend}', "
            f"but the current Keras backend is '{backend}'. Set KERAS_BACKEND={saved_backend} "
            "before importing Keras to load this model."
        )

    if backend == "tensorflow":
        tf = _get_tf()
        if tf is None:
            raise MissingDependencyException(
                "Tensorflow is required in order to load a Keras model using the TensorFlow backend."
            )
        if "GPU" in device_name:
            physical_devices = tf.config.list_physical_devices("GPU")
            if physical_devices:
                try:
                    # an optimization for GPU memory growth. But it will raise an error if any
                    # tensorflow session is already created. That happens when users test runners
                    # in a notebook or Python interactive shell. Thus we just ignore the error.
                    tf.config.experimental.set_memory_growth(physical_devices[0], True)
                except RuntimeError:
                    pass

    device_name = _normalize_device_name(backend, device_name)

    with _device_scope(backend, device_name):
        return keras.models.load_model(
            bento_model.path,
            custom_objects=bento_model.custom_objects,
        )


def save_model(
    name: Tag | str,
    model: "keras.Model",
    *,
    tf_signatures: "tf_ext.ConcreteFunction" | None = None,
    tf_save_options: "tf_ext.SaveOptions" | None = None,
    include_optimizer: bool = False,
    signatures: (
        t.Dict[str, ModelSignature] | t.Dict[str, ModelSignatureDict] | None
    ) = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    external_modules: t.Optional[t.List[ModuleType]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> bentoml.Model:
    """
    Save a model instance to BentoML modelstore.

    ``bentoml.keras.save_model`` works with Keras 2 and Keras 3 models. For
    Keras 3, the active backend (``tensorflow``, ``torch``, ``jax``) is
    detected automatically and recorded in the saved model's context.
    TensorFlow-specific arguments (``tf_signatures`` and
    ``tf_save_options``) are only accepted when the active backend is
    TensorFlow.

    Args:
        name: Name for given model instance. This should pass Python identifier check.
        model: Instance of the Keras model to be saved to BentoML model store.
        tf_signatures: Refer to `Signatures explanation <https://www.tensorflow.org/api_docs/python/tf/saved_model/save>`_
                       from Tensorflow documentation for more information. Only used when the
                       active Keras backend is TensorFlow.
        tf_save_options: :obj:`tf.saved_model.SaveOptions` object that specifies options for saving.
                       Only used when the active Keras backend is TensorFlow.
        signatures: Methods to expose for running inference on the target model. Signatures
                    are used for creating Runner instances when serving model with bentoml.Service
        labels: user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects: Dictionary of Keras custom objects, if specified.
        external_modules: user-defined additional python modules to be saved alongside the model or custom objects,
                          e.g. a tokenizer module, preprocessor module, model configuration module
        metadata: Custom metadata for given model.

    Returns:
        :obj:`~bentoml.Model`: A BentoML model containing the saved Keras model instance.
    """

    if not isinstance(
        model,
        (
            t.cast("t.Type[keras.Model]", LazyType("keras.Model")),
            t.cast("t.Type[keras.Sequential]", LazyType("keras.Sequential")),
        ),
    ):
        raise TypeError(
            f"Given model ({model}) is not a keras.model.Model or keras.engine.sequential.Sequential."
        )

    backend = _get_keras_backend()

    if backend != "tensorflow" and (
        tf_signatures is not None or tf_save_options is not None
    ):
        raise BentoMLException(
            f"'tf_signatures' and 'tf_save_options' are only supported when the Keras backend is TensorFlow. "
            f"Current backend: {backend}."
        )

    context = _get_context(backend)

    if signatures is None:
        signatures = {
            "predict": {
                "batchable": False,
            }
        }
        logger.info(
            'Using the default model signature for Keras (%s) for model "%s".',
            signatures,
            name,
        )

    options = ModelOptions(include_optimizer=include_optimizer)
    kwargs: t.Dict[str, t.Any] = {}
    if tf_signatures is not None:
        kwargs["signatures"] = tf_signatures
    if tf_save_options is not None:
        kwargs["options"] = tf_save_options

    with bentoml.models._create(  # type: ignore
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        options=options,
        context=context,
        labels=labels,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
        signatures=signatures,
    ) as bento_model:
        if version.parse(keras.__version__) >= version.parse("3.4.0"):
            model.save(
                bento_model.path,
                zipped=False,
                include_optimizer=include_optimizer,
                **kwargs,
            )
        else:
            model.save(bento_model.path, include_optimizer=include_optimizer, **kwargs)

        return bento_model


def _to_numpy(value: t.Any) -> t.Any:
    """Convert backend tensors returned by Keras methods to numpy arrays."""
    if LazyType("torch.Tensor").isinstance(value):
        return value.detach().cpu().numpy()
    if LazyType("jax.Array").isinstance(value):
        import numpy as np

        return np.asarray(value)
    if isinstance(value, (list, tuple)):
        return tuple(_to_numpy(v) for v in value)
    return value


def get_runnable(
    bento_model: bentoml.Model,
):
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    partial_kwargs: t.Dict[str, t.Any] = bento_model.info.options.partial_kwargs  # type: ignore

    backend = _get_keras_backend()

    class KerasRunnable(Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()
            self.device_name = _normalize_device_name(
                backend, _get_default_device(backend)
            )
            self.model = load_model(bento_model, device_name=self.device_name)
            self.methods_cache: t.Dict[str, t.Callable[..., t.Any]] = {}

    if backend == "tensorflow":

        def _gen_run_method(runnable_self: KerasRunnable, method_name: str):
            raw_method = getattr(runnable_self.model, method_name)
            method_partial_kwargs = partial_kwargs.get(method_name)
            if method_partial_kwargs:
                raw_method = functools.partial(raw_method, **method_partial_kwargs)

            tf = _get_tf()
            if tf is None:
                raise MissingDependencyException(
                    "Tensorflow is required in order to run module 'bentoml.keras' with the TensorFlow backend."
                )

            def _mapping(item: "KerasArgType") -> "tf_ext.TensorLike":
                if not LazyType["tf_ext.TensorLike"]("tensorflow.Tensor").isinstance(
                    item
                ):
                    return t.cast("tf_ext.TensorLike", tf.convert_to_tensor(item))
                else:
                    return item

            def _run_method(
                runnable_self: KerasRunnable, *args: "KerasArgType"
            ) -> "ext.NpNDArray" | t.Tuple["ext.NpNDArray", ...]:
                params = Params["KerasArgType"](*args)

                with tf.device(runnable_self.device_name):
                    params = params.map(_mapping)
                    if len(params.args) == 1:
                        arg = params.args[0]
                    else:
                        arg = params.args

                    res: "tf_ext.EagerTensor" | "ext.NpNDArray" = raw_method(arg)

                    if LazyType["tf_ext.EagerTensor"](
                        "tensorflow.python.framework.ops._EagerTensorBase"
                    ).isinstance(res):
                        return t.cast("ext.NpNDArray", res.numpy())

                    if isinstance(res, list):
                        return tuple(res)
                    return res

            return _run_method

    elif backend == "torch":

        def _gen_run_method(runnable_self: KerasRunnable, method_name: str):
            import torch

            raw_method = getattr(runnable_self.model, method_name)
            method_partial_kwargs = partial_kwargs.get(method_name)
            if method_partial_kwargs:
                raw_method = functools.partial(raw_method, **method_partial_kwargs)

            def _mapping(item: t.Any) -> t.Any:
                if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(item):
                    return torch.from_numpy(item).to(runnable_self.device_name)
                if LazyType("torch.Tensor").isinstance(item):
                    return item.to(runnable_self.device_name)
                if isinstance(item, (list, tuple)):
                    return torch.as_tensor(item).to(runnable_self.device_name)
                return item

            def _run_method(runnable_self: KerasRunnable, *args: t.Any) -> t.Any:
                params = Params(*args)
                params = params.map(_mapping)
                if len(params.args) == 1:
                    arg = params.args[0]
                else:
                    arg = params.args

                res = raw_method(arg)
                return _to_numpy(res)

            return _run_method

    else:

        def _gen_run_method(runnable_self: KerasRunnable, method_name: str):
            raw_method = getattr(runnable_self.model, method_name)
            method_partial_kwargs = partial_kwargs.get(method_name)
            if method_partial_kwargs:
                raw_method = functools.partial(raw_method, **method_partial_kwargs)

            def _run_method(runnable_self: KerasRunnable, *args: t.Any) -> t.Any:
                params = Params(*args)
                params = params.map(_to_numpy)
                if len(params.args) == 1:
                    arg = params.args[0]
                else:
                    arg = params.args

                res = raw_method(arg)
                return _to_numpy(res)

            return _run_method

    def add_run_method(method_name: str, options: ModelSignature):
        def run_method(runnable_self: KerasRunnable, *args: t.Any) -> t.Any:
            _run_method = runnable_self.methods_cache.get(method_name)
            if not _run_method:
                _run_method = _gen_run_method(runnable_self, method_name)
                runnable_self.methods_cache[method_name] = _run_method

            return _run_method(runnable_self, *args)

        KerasRunnable.add_method(
            run_method,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_run_method(method_name, options)

    return KerasRunnable
