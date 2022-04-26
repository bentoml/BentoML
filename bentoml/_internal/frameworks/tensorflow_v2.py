import os
import uuid
import typing as t
import logging
import functools
from typing import TYPE_CHECKING

import bentoml
from bentoml import Tag
from bentoml import Model as BentoModel
from bentoml import Runnable
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..types import LazyType
from ..runner.utils import Params
from ..utils.tensorflow import get_tf_version
from ..utils.tensorflow import is_gpu_available
from ..utils.tensorflow import hook_loaded_model

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf  # type: ignore
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
    `tensorflow` is required in order to use `bentoml.tensorflow`.
    Instruction: `pip install tensorflow`
    """
    )

try:
    import tensorflow_hub as hub  # type: ignore
    from tensorflow_hub import resolve  # type: ignore
    from tensorflow_hub import native_module  # type: ignore
except ImportError:  # pragma: no cover
    logger.warning(
        """\
    If you want to use `bentoml.tensorflow.import_from_tfhub(),
     make sure to `pip install --upgrade tensorflow_hub` before using.
     """
    )
    hub = None


try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata


if TYPE_CHECKING:
    from tensorflow_hub import Module as HubModule  # type: ignore
    from tensorflow_hub import KerasLayer  # type: ignore

    from .. import external_typing as ext
    from ..types import PathType
    from ..models import ModelStore
    from ..external_typing import tensorflow as tf_ext

    TFArgType = t.Union[t.List[t.Union[int, float]], ext.NpNDArray, tf_ext.Tensor]

MODULE_NAME = "bentoml.tensorflow_v2"


def _clean_name(name: str) -> str:  # pragma: no cover
    if name.startswith(("http://", "https://")):
        name = name.split("/", maxsplit=3)[-1]
    else:
        name = name.split("/")[-1]
    return re.sub(r"\W|^(?=\d)-", "_", name)


def _load_model(
    bentomodel: BentoModel,
    tags: t.Optional[t.List[str]] = None,
    load_as_hub_module: t.Optional[bool] = None,
) -> t.Union["tf_ext.AutoTrackable", "tf_ext.Module", "HubModule", "KerasLayer"]:

    if bentomodel.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {bentomodel.tag} was saved with module {bentomodel.info.module}, failed loading with {MODULE_NAME}."
        )
    if bentomodel.info.context["import_from_tfhub"]:
        assert load_as_hub_module is not None, (
            "You have to specified `load_as_hub_module=True | False`"
            " to load a `tensorflow_hub` module. If True is chosen,"
            " then BentoML will return either an instance of `hub.KerasLayer`"
            " or `hub.Module` depending on your TF version. For most usecase,"
            " we recommend to keep `load_as_hub_module=True`. If you wish to extend"
            " the functionalities of the given model, set `load_as_hub_module=False`"
            " will return a SavedModel object."
        )

        if hub is None:
            raise MissingDependencyException(
                """\
            `tensorflow_hub` does not exists.
            Make sure to `pip install --upgrade tensorflow_hub` before using.
            """
            )

        module_path = bentomodel.path_of(bentomodel.info.options["local_path"])
        if load_as_hub_module:
            return (
                hub.Module(module_path)
                if get_tf_version().startswith("1")
                else hub.KerasLayer(module_path)
            )
        # In case users want to load as a SavedModel file object.
        # https://github.com/tensorflow/hub/blob/master/tensorflow_hub/module_v2.py#L93
        is_hub_module_v1: bool = tf.io.gfile.exists(  # type: ignore
            native_module.get_module_proto_path(module_path)
        )
        if tags is None and is_hub_module_v1:
            tags = []
        tf_model: "tf_ext.AutoTrackable" = tf.saved_model.load(  # type: ignore
            module_path,
            tags=tags,
        )
        tf_model._is_hub_module_v1 = (
            is_hub_module_v1  # pylint: disable=protected-access # noqa
        )
        return tf_model
    else:
        tf_model: "tf_ext.AutoTrackable" = tf.saved_model.load(model.path)  # type: ignore
        return hook_loaded_model(tf_model, MODULE_NAME)


def load_model(
    bentomodel: t.Union[str, Tag, BentoModel],
    tags: t.Optional[t.List[str]] = None,
    load_as_hub_module: t.Optional[bool] = None,
    device_id: str = "/device/CPU:0",
) -> t.Union["tf_ext.AutoTrackable", "tf_ext.Module", "HubModule", "KerasLayer"]:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        bento_tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        tags (:code:`str`, `optional`, defaults to `None`):
            A set of strings specifying the graph variant to use, if loading from a v1 module.
        options (:code:`tensorflow.saved_model.SaveOptions`, `optional`, default to :code:`None`):
            :code:`tensorflow.saved_model.LoadOptions` object that specifies options for loading. This
            argument can only be used from TensorFlow 2.3 onwards.
        load_as_hub_module (`bool`, `optional`, default to :code:`True`):
            Load the given weight that is saved from tfhub as either `hub.KerasLayer` or `hub.Module`.
            The latter only applies for TF1.

    Returns:
        :obj:`SavedModel`: an instance of :obj:`SavedModel` format from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        # load a model back into memory
        model = bentoml.tensorflow.load("my_tensorflow_model")

    """  # noqa: LN001
    if not isinstance(bentomodel, BentoModel):
        bentomodel = bentoml.models.get(bentomodel)

    with tf.device(device_id):
        return _load_model(bentomodel, tags=tags, load_as_hub_module=load_as_hub_module)


def import_from_tfhub(
    identifier: t.Union[str, "HubModule", "KerasLayer"],
    name: t.Optional[str] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> Tag:
    """
    Import a model from `Tensorflow Hub <https://tfhub.dev/>`_ to BentoML modelstore.

    Args:
        identifier (:code:`Union[str, tensorflow_hub.Module, tensorflow_hub.KerasLayer]`): Identifier accepts
            two type of inputs:
                - if `type` of :code:`identifier` either of type :code:`tensorflow_hub.Module` (**legacy** `tensorflow_hub`) or :code:`tensorflow_hub.KerasLayer` (`tensorflow_hub`), then we will save the given model to a :code:`SavedModel` format.
                - if `type` of :code:`identifier` is a :obj:`str`, we assume that this is the URI retrieved from Tensorflow Hub. We then clean the given URI, and get a local copy of a given model to BentoML modelstore. name (:code:`str`, `optional`, defaults to `None`): An optional name for the model. If :code:`identifier` is a :obj:`str`, then name can be autogenerated from the given URI.
        name (:code:`str`, `optional`, default to `None`):
            Optional name for the saved model. If None, then name will be generated from :code:`identifier`.
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`~bentoml.Tag` object that can be used to retrieve the model with :func:`bentoml.tensorflow.load`:

    Example for importing a model from Tensorflow Hub:

    .. code-block:: python

        import tensorflow_text as text # noqa # pylint: disable
        import bentoml

        tag = bentoml.tensorflow.import_from_tfhub("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

        # load model back with `load`:
        model = bentoml.tensorflow.load(tag, load_as_hub_module=True)


    Example for importing a custom Tensorflow Hub model:

    .. code-block:: python

        import tensorflow as tf
        import tensorflow_hub as hub
        import bentoml

        def _plus_one_model_tf2():
            obj = tf.train.Checkpoint()

            @tf.function(input_signature=[tf.TensorSpec(None, dtype=tf.float32)])
            def plus_one(x):
                return x + 1

            obj.__call__ = plus_one
            return obj

        # then save the given model to BentoML modelstore:
        model = _plus_one_model_tf2()
        tag = bentoml.tensorflow.import_from_tfhub(model)
    """  # noqa
    if hub is None:
        raise MissingDependencyException(
            """\
        `tensorflow_hub` does not exists.
        Make sure to `pip install --upgrade tensorflow_hub` before using.
        """
        )
    context: t.Dict[str, t.Any] = {
        "framework_name": "tensorflow",
        "pip_dependencies": [
            f"tensorflow=={get_tf_version()}",
            f"tensorflow_hub=={importlib_metadata.version('tensorflow_hub')}",
        ],
        "import_from_tfhub": True,
    }
    if name is None:
        if isinstance(identifier, str):
            name = _clean_name(identifier)
        else:
            name = f"{identifier.__class__.__name__}_{uuid.uuid4().hex[:5].upper()}"

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        options=None,
        context=context,
        metadata=metadata,
        labels=labels,
        custom_objects=custom_objects,
    ) as _model:
        if isinstance(identifier, str):
            current_cache_dir = os.environ.get("TFHUB_CACHE_DIR")
            os.environ["TFHUB_CACHE_DIR"] = _model.path
            fpath: str = resolve(identifier)
            folder = fpath.split("/")[-1]
            _model.info.options = {"model": identifier, "local_path": folder}
            if current_cache_dir is not None:
                os.environ["TFHUB_CACHE_DIR"] = current_cache_dir
        else:
            if hasattr(identifier, "export"):
                # hub.Module.export()
                with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph()) as sess:  # type: ignore
                    sess.run(tf.compat.v1.global_variables_initializer())  # type: ignore
                    identifier.export(_model.path, sess)  # type: ignore
            else:
                tf.saved_model.save(identifier, _model.path)
            _model.info.options = {
                "model": identifier.__class__.__name__,
                "local_path": ".",
            }

        return _model.tag


def save_model(
    name: str,
    model: t.Union["tf_ext.KerasModel", "tf_ext.Module"],
    *,
    tf_signatures: t.Optional["tf_ext.ConcreteFunction"] = None,
    tf_save_options: t.Optional["tf_ext.SaveOptions"] = None,
    signatures: t.Optional[t.Dict[str, t.Any]],
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> Tag:

    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[keras.Model, tf.Module]`):
            Instance of model to be saved
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        signatures (:code:`Union[Callable[..., Any], dict]`, `optional`, default to :code:`None`):
            Refers to `Signatures explanation <https://www.tensorflow.org/api_docs/python/tf/saved_model/save>`_
            from Tensorflow documentation for more information.
        tf_save_options (`tf.saved_model.SaveOptions`, `optional`, default to :code:`None`):
            :obj:`tf.saved_model.SaveOptions` object that specifies options for saving.

    Raises:
        ValueError: If :obj:`obj` is not trackable.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import tensorflow as tf
        import numpy as np
        import bentoml

        class NativeModel(tf.Module):
            def __init__(self):
                super().__init__()
                self.weights = np.asfarray([[1.0], [1.0], [1.0], [1.0], [1.0]])
                self.dense = lambda inputs: tf.matmul(inputs, self.weights)

            @tf.function(
                input_signature=[tf.TensorSpec(shape=[1, 5], dtype=tf.float64, name="inputs")]
            )
            def __call__(self, inputs):
                return self.dense(inputs)

        # then save the given model to BentoML modelstore:
        model = NativeModel()
        tag = bentoml.tensorflow.save_model("native_toy", model)

    .. note::

       :code:`bentoml.tensorflow.save_model` API also support saving `RaggedTensor <https://www.tensorflow.org/guide/ragged_tensor>`_ model and Keras model. If you choose to save a Keras model
       with :code:`bentoml.tensorflow.save_model`, then the model will be saved under a :obj:`SavedModel` format instead of :obj:`.h5`.

    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "tensorflow",
        "pip_dependencies": [f"tensorflow=={get_tf_version()}"],
        "import_from_tfhub": False,
    }
    if signatures is None:
        signatures = {
            "__call__": dict(
                batchable=True,
                batch_dim=0,
            )
        }
    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        options=None,
        context=context,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
        signatures=signatures,
    ) as _model:

        tf.saved_model.save(
            model,
            _model.path,
            signatures=tf_signatures,
            options=tf_save_options,
        )

        return _model.tag


def _to_runner(
    bentomodel: BentoModel,
    tags: t.Optional[t.List[str]] = None,
    load_as_hub_module: t.Optional[bool] = None,
    partial_kwargs: t.Dict[str, Any] = None,
    **kwargs,
):
    class TensorflowRunnable(Runnable):
        SUPPORT_GPU: True
        SUPPORT_MULTI_THREADING: True

        def __init__(self):
            super().__init__(**kwargs)
            if tf.list_physical_devices("GPU") > 0:
                self.device_id = "/device/GPU:0"
            else:
                self.device_id = "/device/CPU:0"
                self.num_threads = int(os.environ["BENTOML_NUM_THREAD"])
                tf.config.threading.set_inter_op_parallelism_threads(self.num_threads)
                tf.config.threading.set_intra_op_parallelism_threads(self.num_threads)

            self.model = bentomodel.load_model(
                tags=tags, load_as_hub_module=load_as_hub_module, device_id=device_id
            )

            partial_kwargs = partial_kwargs or dict()
            for method_name, signature in bentomodel.signatures.items():
                raw_method = getattr(self.model, method_name)
                method_partial_kwargs = partial_kwargs.get(method_name)
                if method_partial_kwargs:
                    raw_method = functools.partial(raw_method, **method_partial_kwargs)

                def _method_func(runnable_self, *args, **kwargs):
                    params = Params["TFArgType"](*args, **kwargs)

                    with tf.device(runnable_self.device_id):  # type: ignore

                        def _mapping(item: "TFArgType") -> "tf_ext.TensorLike":
                            if not LazyType["tf_ext.TensorLike"](
                                "tf.Tensor"
                            ).isinstance(item):
                                return t.cast(
                                    "tf_ext.TensorLike", tf.convert_to_tensor(item)
                                )
                            else:
                                return item

                        params = params.map(_mapping)

                        tf.compat.v1.global_variables_initializer()  # type: ignore
                        res = raw_method(*params.args, **params.kwargs)
                        return t.cast("ext.NpNDArray", res.numpy())

                deco_method = Runnable.method(**signatures)(_method_func)
                setattr(TensorflowRunnable, method_name, deco_method)

    return TensorflowRunnable
