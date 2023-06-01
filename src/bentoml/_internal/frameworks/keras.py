from __future__ import annotations

import typing as t
import logging
import functools
from types import ModuleType
from typing import TYPE_CHECKING

import attr

import bentoml
from bentoml import Tag
from bentoml import Runnable
from bentoml.models import ModelContext
from bentoml.exceptions import NotFound
from bentoml.exceptions import MissingDependencyException

from ..types import LazyType
from ..models.model import ModelSignature
from ..models.model import PartialKwargsModelOptions
from ..runner.utils import Params
from .utils.tensorflow import get_tf_version

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from .. import external_typing as ext
    from ..models.model import ModelSignatureDict
    from ..external_typing import tensorflow as tf_ext

    KerasArgType = t.Union[t.List[t.Union[int, float]], ext.NpNDArray, tf_ext.Tensor]

try:
    import keras
    import tensorflow as tf
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "Tensorflow is required in order to use module 'bentoml.keras', since BentoML uses Tensorflow as Keras backend. Install Tensorflow with 'pip install tensorflow'. For more information, refer to https://www.tensorflow.org/install"
    )

MODULE_NAME = "bentoml.keras"
API_VERSION = "v1"


@attr.define
class KerasOptions(PartialKwargsModelOptions):
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


def load_model(
    bento_model: str | Tag | bentoml.Model,
    device_name: str = "/device:CPU:0",
) -> "tf_ext.KerasModel":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        bento_model (``str`` ``|`` :obj:`~bentoml.Tag` ``|`` :obj:`~bentoml.Model`):
            Either the tag of the model to get from the store, or a BentoML `~bentoml.Model`
            instance to load the model from.
        device_name (``str`` | ``None``):
            The device id to load the model on. The device id format should be compatible with `tf.device <https://www.tensorflow.org/api_docs/python/tf/device>`_


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

    if "GPU" in device_name:
        physical_devices = tf.config.list_physical_devices("GPU")
        try:
            # an optimization for GPU memory growth. But it will raise an error if any
            # tensorflow session is already created. That happens when users test runners
            # in a notebook or Python interactive shell. Thus we just ignore the error.
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError:
            pass

    with tf.device(device_name):
        return keras.models.load_model(
            bento_model.path,
            custom_objects=bento_model.custom_objects,
        )


def save_model(
    name: Tag | str,
    model: "tf_ext.KerasModel",
    *,
    tf_signatures: "tf_ext.ConcreteFunction" | None = None,
    tf_save_options: "tf_ext.SaveOptions" | None = None,
    include_optimizer: bool = False,
    signatures: t.Dict[str, ModelSignature]
    | t.Dict[str, ModelSignatureDict]
    | None = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    external_modules: t.Optional[t.List[ModuleType]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> bentoml.Model:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name: Name for given model instance. This should pass Python identifier check.
        model: Instance of the Keras model to be saved to BentoML model store.
        tf_signatures: Refer to `Signatures explanation <https://www.tensorflow.org/api_docs/python/tf/saved_model/save>`_
                       from Tensorflow documentation for more information.
        tf_save_options: :obj:`tf.saved_model.SaveOptions` object that specifies options for saving.
        signatures: Methods to expose for running inference on the target model. Signatures
                    are used for creating Runner instances when serving model with bentoml.Service
        labels: user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects: Dictionary of Keras custom objects, if specified.
        external_modules: user-defined additional python modules to be saved alongside the model or custom objects,
                          e.g. a tokenizer module, preprocessor module, model configuration module
        metadata: Custom metadata for given model.

    Returns:
        :obj:`~bentoml.Model`: A BentoML model containing the saved Keras model instance.

    Examples:

    .. code-block:: python

        import bentoml
        import tensorflow as tf
        import tensorflow.keras as keras


        def custom_activation(x):
            return tf.nn.tanh(x) ** 2


        class CustomLayer(keras.layers.Layer):
            def __init__(self, units=32, **kwargs):
                super(CustomLayer, self).__init__(**kwargs)
                self.units = tf.Variable(units, name="units")

            def call(self, inputs, training=False):
                if training:
                    return inputs * self.units
                else:
                    return inputs

            def get_config(self):
                config = super(CustomLayer, self).get_config()
                config.update({"units": self.units.numpy()})
                return config


        def KerasSequentialModel() -> keras.models.Model:
            net = keras.models.Sequential(
                (
                    keras.layers.Dense(
                        units=1,
                        input_shape=(5,),
                        use_bias=False,
                        kernel_initializer=keras.initializers.Ones(),
                    ),
                )
            )

            opt = keras.optimizers.Adam(0.002, 0.5)
            net.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
            return net

        model = KerasSequentialModel()

        # `save` a given model and retrieve coresponding tag:
        bento_model = bentoml.keras.save_model("keras_model", model)

        # `save` a given model with custom objects definition:
        custom_objects = {
            "CustomLayer": CustomLayer,
            "custom_activation": custom_activation,
        },
        custom_bento_model = bentoml.keras.save_model("custom_obj_keras", custom_objects=custom_objects)
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

    context = ModelContext(
        framework_name="keras", framework_versions={"tensorflow": get_tf_version()}
    )

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

    options = KerasOptions(include_optimizer=include_optimizer)

    with bentoml.models.create(
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
        model.save(
            bento_model.path,
            signatures=tf_signatures,
            options=tf_save_options,
            include_optimizer=include_optimizer,
        )

        return bento_model


def get_runnable(
    bento_model: bentoml.Model,
):
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    partial_kwargs: t.Dict[str, t.Any] = bento_model.info.options.partial_kwargs  # type: ignore

    class KerasRunnable(Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()
            if len(tf.config.list_physical_devices("GPU")) > 0:
                # In Multi-GPU scenarios, the visible cuda devices will be set for each Runner worker
                # by the runner's Scheduling Strategy. So that the Runnable implementation only needs
                # to find the first GPU device visible to current process.
                self.device_name = "/device:GPU:0"
            else:
                self.device_name = "/device:CPU:0"

            self.model = load_model(bento_model, device_name=self.device_name)
            self.methods_cache: t.Dict[str, t.Callable[..., t.Any]] = {}

    def _gen_run_method(runnable_self: KerasRunnable, method_name: str):
        raw_method = getattr(runnable_self.model, method_name)
        method_partial_kwargs = partial_kwargs.get(method_name)
        if method_partial_kwargs:
            raw_method = functools.partial(raw_method, **method_partial_kwargs)

        def _mapping(item: "KerasArgType") -> "tf_ext.TensorLike":
            if not LazyType["tf_ext.TensorLike"]("tensorflow.Tensor").isinstance(item):
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

    def add_run_method(method_name: str, options: ModelSignature):
        def run_method(
            runnable_self: KerasRunnable,
            *args: "KerasArgType",
        ) -> "ext.NpNDArray":
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
