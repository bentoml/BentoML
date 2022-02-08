import os
import re
import uuid
import typing as t
import logging
import pathlib
import functools
from typing import TYPE_CHECKING
from distutils.dir_util import copy_tree

from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml import Runner
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..types import LazyType
from ..runner.utils import Params
from ..utils.tensorflow import get_tf_version
from ..utils.tensorflow import is_gpu_available
from ..utils.tensorflow import tf_function_wrapper
from ..utils.tensorflow import pretty_format_restored_model
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.python.client import device_lib
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
    `tensorflow` is required in order to use `bentoml.tensorflow`.
    Instruction: `pip install tensorflow`
    """
    )

try:
    import tensorflow_hub as hub
    from tensorflow_hub import resolve
    from tensorflow_hub import native_module
except ImportError:  # pragma: no cover
    logger.warning(
        """\
    If you want to use `bentoml.tensorflow.import_from_tfhub(),
     make sure to `pip install --upgrade tensorflow_hub` before using.
     """
    )
    hub, resolve, native_module = None, None, None


try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata


if TYPE_CHECKING:
    from .. import external_typing as ext
    from ..types import PathType
    from ..models import ModelStore
    from ..external_typing import tensorflow as tf_ext

    from tensorflow_hub import KerasLayer, Module

    TFArgType = t.Union[t.List[t.Union[int, float]], "ext.NpNDArray", "tf_ext.Tensor"]

MODULE_NAME = "bentoml.tensorflow"


def _clean_name(name: str) -> str:  # pragma: no cover
    return re.sub(r"\W|^(?=\d)-", "_", name)


AUTOTRACKABLE_CALLABLE_WARNING: str = """\
Importing SavedModels from TensorFlow 1.x. `outputs = imported(inputs)`
 will not be supported by BentoML due to `tensorflow` API.\n
See https://www.tensorflow.org/api_docs/python/tf/saved_model/load for
 more details.
"""

TF_FUNCTION_WARNING: str = """\
Due to TensorFlow's internal mechanism, only methods
 wrapped under `@tf.function` decorator and the Keras default function
 `__call__(inputs, training=False)` can be restored after a save & load.\n
You can test the restored model object via `bentoml.tensorflow.load(path)`
"""

KERAS_MODEL_WARNING: str = """\
BentoML detected that {name} is being used to pack a Keras API
 based model. In order to get optimal serving performance, we recommend
 to wrap your keras model `call()` methods with `@tf.function` decorator.
"""


@inject
def load(
    tag: Tag,
    tfhub_tags: t.Optional[t.List[str]] = None,
    tfhub_options: t.Optional["tf_ext.SaveOptions"] = None,
    load_as_wrapper: t.Optional[bool] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Union["tf_ext.AutoTrackable", "KerasLayer", "Module"]:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        tfhub_tags (:code:`str`, `optional`, defaults to `None`):
            A set of strings specifying the graph variant to use, if loading from a v1 module.
        tfhub_options (:code:`tensorflow.saved_model.SaveOptions`, `optional`, default to :code:`None`):
            :code:`tensorflow.saved_model.LoadOptions` object that specifies options for loading. This
            argument can only be used from TensorFlow 2.3 onwards.
        load_as_wrapper (`bool`, `optional`, default to :code:`True`):
            Load the given weight that is saved from tfhub as either `hub.KerasLayer` or `hub.Module`.
            The latter only applies for TF1.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`SavedModel`: an instance of :obj:`SavedModel` format from BentoML modelstore.

    Examples:

    .. tabs::

        .. code-tab:: tensorflow_v1

            import bentoml

            # load a model back into memory
            model = bentoml.tensorflow.load("my_tensorflow_model")

        .. code-tab:: tensorflow_v2

            import bentoml

            # load a model back into memory
            model = bentoml.tensorflow.load("my_tensorflow_model")

    """  # noqa: LN001
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    if model.info.context["import_from_tfhub"]:
        assert load_as_wrapper is not None, (
            "You have to specified `load_as_wrapper=True | False`"
            " to load a `tensorflow_hub` module. If True is chosen,"
            " then BentoML will return either an instance of `hub.KerasLayer`"
            " or `hub.Module` depending on your TF version. For most usecase,"
            " we recommend to keep `load_as_wrapper=True`. If you wish to extend"
            " the functionalities of the given model, set `load_as_wrapper=False`"
            " will return a SavedModel object."
        )

        if hub is None:
            raise MissingDependencyException(
                """\
            `tensorflow_hub` does not exists.
            Make sure to `pip install --upgrade tensorflow_hub` before using.
            """
            )

        module_path = model.path_of(model.info.options["local_path"])
        if load_as_wrapper:
            return (
                hub.KerasLayer(module_path)
                if get_tf_version().startswith("2")
                else hub.Module(module_path)
            )
        # In case users want to load as a SavedModel file object.
        # https://github.com/tensorflow/hub/blob/master/tensorflow_hub/module_v2.py#L93
        is_hub_module_v1: bool = tf.io.gfile.exists(
            native_module.get_module_proto_path(module_path)
        )
        if tfhub_tags is None and is_hub_module_v1:
            tfhub_tags = []
        if tfhub_options is not None:
            if not LazyType("tf.saved_model.SaveOptions").isinstance(tfhub_options):
                raise BentoMLException(
                    f"`tfhub_options` has to be of type `tf.saved_model.SaveOptions`, got {type(tfhub_options)} instead."
                )
            if not hasattr(getattr(tf, "saved_model", None), "LoadOptions"):
                raise NotImplementedError(
                    "options are not supported for TF < 2.3.x,"
                    f" Current version: {get_tf_version()}"
                )
            tf_model = tf.saved_model.load(
                module_path, tags=tfhub_tags, options=tfhub_options
            )
        else:
            tf_model = tf.saved_model.load(module_path, tags=tfhub_tags)
        tf_model._is_hub_module_v1 = (
            is_hub_module_v1  # pylint: disable=protected-access # noqa
        )
        return tf_model
    else:
        tf_model: "tf_ext.AutoTrackable" = tf.saved_model.load(model.path)
        tf_function_wrapper.hook_loaded_model(tf_model)
        logger.warning(TF_FUNCTION_WARNING)
        # pretty format loaded model
        logger.info(pretty_format_restored_model(tf_model))
        if hasattr(tf_model, "keras_api"):
            logger.warning(KERAS_MODEL_WARNING.format(name=MODULE_NAME))
        return tf_model


@inject
def import_from_tfhub(
    identifier: t.Union[str, "tf.HubModule", "tf.KerasLayer"],
    name: t.Optional[str] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Import a model from `Tensorflow Hub <https://tfhub.dev/>`_ to BentoML modelstore.

    Args:
        identifier (:code:`Union[str, tensorflow_hub.Module, tensorflow_hub.KerasLayer]`): Identifier accepts
            two type of inputs:
                - if `type` of :code:`identifier` either of type :code:`tensorflow_hub.Module` (**legacy** `tensorflow_hub`) or :code:`tensorflow_hub.KerasLayer` (`tensorflow_hub`), then we will save the given model to a :code:`SavedModel` format.
                - if `type` of :code:`identifier` is a :obj:`str`, we assume that this is the URI retrieved from Tensorflow Hub. We then clean the given URI, and get a local copy of a given model to BentoML modelstore. name (:code:`str`, `optional`, defaults to `None`): An optional name for the model. If :code:`identifier` is a :obj:`str`, then name can be autogenerated from the given URI.
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`~bentoml._internal.types.Tag` object that can be used to retrieve the model with :func:`bentoml.tensorflow.load`:

    Example for importing a model from Tensorflow Hub:

    .. code-block:: python

        import tensorflow_text as text # noqa # pylint: disable
        import bentoml

        tag = bentoml.tensorflow.import_from_tfhub("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

        # load model back with `load`:
        model = bentoml.tensorflow.load(tag, load_as_wrapper=True)

    Example for importing a custom Tensorflow Hub model:

    .. tabs::

        .. code-tab:: tensorflow_v1

            import tensorflow as tf
            import tensorflow_hub as hub
            import bentoml

            # Simple toy Tensorflow Hub model
            def _plus_one_model_tf1():
                def plus_one():
                    x = tf.compat.v1.placeholder(dtype=tf.float32, name="x")
                    y = x + 1
                    hub.add_signature(inputs=x, outputs=y)

                spec = hub.create_module_spec(plus_one)
                with tf.compat.v1.get_default_graph().as_default():
                    module = hub.Module(spec, trainable=True)
                    return module

            model = _plus_one_model_tf1()

            # retrieve the given tag:
            tag = bentoml.tensorflow.import_from_tfhub(model)

        .. code-tab:: tensorflow_v2

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
    if resolve is None:
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
            if identifier.startswith(("http://", "https://")):
                name = _clean_name(identifier.split("/", maxsplit=3)[-1])
            else:
                name = _clean_name(identifier.split("/")[-1])
        else:
            name = f"{identifier.__class__.__name__}_{uuid.uuid4().hex[:5].upper()}"
    _model = Model.create(
        name,
        module=MODULE_NAME,
        options=None,
        context=context,
        metadata=metadata,
    )
    if isinstance(identifier, str):
        os.environ["TFHUB_CACHE_DIR"] = _model.path
        fpath = resolve(identifier)
        folder = fpath.split("/")[-1]
        _model.info.options = {"model": identifier, "local_path": folder}
    else:
        if hasattr(identifier, "export"):
            # hub.Module.export()
            with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph()) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                identifier.export(_model.path, sess)
        else:
            tf.saved_model.save(identifier, _model.path)
        _model.info.options = {
            "model": identifier.__class__.__name__,
            "local_path": ".",
        }

    _model.save(model_store)
    return _model.tag


@inject
def save(
    name: str,
    model: t.Union["PathType", "tf.KerasModel", "tf.Module"],
    *,
    signatures: t.Optional["tf.ConcreteFunction"] = None,
    options: t.Optional["tf.saved_model.SaveOptions"] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[keras.Model, tf.Module, path-like objects]`):
            Instance of model to be saved
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
        signatures (:code:`Union[Callable[..., Any], dict]`, `optional`, default to :code:`None`):
            Refers to `Signatures explanation <https://www.tensorflow.org/api_docs/python/tf/saved_model/save>`_
            from Tensorflow documentation for more information.
        options (`tf.saved_model.SaveOptions`, `optional`, default to :code:`None`):
            :obj:`tf.saved_model.SaveOptions` object that specifies options for saving.

    Raises:
        ValueError: If :obj:`obj` is not trackable.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. tabs::

        .. code-tab:: tensorflow_v1

            import tensorflow as tf
            import tempfile
            import bentoml

            location = ""

            # Function below builds model graph
            def cnn_model_fn():
                X = tf.compat.v1.placeholder(shape=[None, 2], dtype=tf.float32, name="X")

                # dense layer
                inter1 = tf.compat.v1.layers.dense(inputs=X, units=1, activation=tf.nn.relu)
                p = tf.argmax(input=inter1, axis=1)

                # loss
                y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="y")
                loss = tf.losses.softmax_cross_entropy(y, inter1)

                # training operation
                train_op = tf.compat.v1.train.AdamOptimizer().minimize(loss)

                return {"p": p, "loss": loss, "train_op": train_op, "X": X, "y": y}

            cnn_model = cnn_model_fn()

            with tempfile.TemporaryDirectory() as temp_dir:
                with tf.compat.v1.Session() as sess:
                    sess.run(tf.compat.v1.global_variables_initializer())
                    sess.run(cnn_model["p"], {cnn_model["X"]: test_data})

                    inputs = {"X": cnn_model["X"]}
                    outputs = {"prediction": cnn_model["p"]}

                    tf.compat.v1.saved_model.simple_save(
                        sess, temp_dir, inputs=inputs, outputs=outputs
                    )
                    location = temp_dir

            # then save the given model to BentoML modelstore:
            tag = bentoml.tensorflow.save("cnn_model", location)

        .. code-tab:: tensorflow_v2

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
            tag = bentoml.tensorflow.save("native_toy", model)

    .. note::

       :code:`bentoml.tensorflow.save` API also support saving `RaggedTensor <https://www.tensorflow.org/guide/ragged_tensor>`_ model and Keras model. If you choose to save a Keras model
       with :code:`bentoml.tensorflow.save`, then the model will be saved under a :obj:`SavedModel` format instead of :obj:`.h5`.

    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "tensorflow",
        "pip_dependencies": [f"tensorflow=={get_tf_version()}"],
        "import_from_tfhub": False,
    }
    _model = Model.create(
        name,
        module=MODULE_NAME,
        options=None,
        context=context,
        metadata=metadata,
    )

    if isinstance(model, (str, bytes, os.PathLike, pathlib.Path)):  # type: ignore[reportUnknownMemberType] # noqa
        assert os.path.isdir(model)
        copy_tree(str(model), _model.path)
    else:
        tf.saved_model.save(model, _model.path, signatures=signatures, options=options)

    _model.save(model_store)
    return _model.tag


class _TensorflowRunner(Runner):
    @inject
    def __init__(
        self,
        tag: Tag,
        predict_fn_name: str,
        device_id: str,
        partial_kwargs: t.Dict[str, t.Any],
        name: str,
        resource_quota: t.Dict[str, t.Any],
        batch_options: t.Dict[str, t.Any],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        in_store_tag = model_store.get(tag).tag
        self._tag = in_store_tag
        super().__init__(name, resource_quota, batch_options)

        self._device_id = device_id
        self._configure(device_id)
        self._predict_fn_name = predict_fn_name
        assert any(device_id in d.name for d in device_lib.list_local_devices())
        self._partial_kwargs: t.Dict[str, t.Any] = (
            partial_kwargs if partial_kwargs is not None else dict()
        )
        self._model_store = model_store

    def _configure(self, device_id: str) -> None:
        if "GPU" in device_id:
            tf.config.set_visible_devices(device_id, "GPU")
        self._config_proto = dict(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=self._num_threads,
            inter_op_parallelism_threads=self._num_threads,
        )

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._tag]

    @property
    def _num_threads(self) -> int:
        if is_gpu_available() and self.resource_quota.on_gpu:
            return 1
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        if is_gpu_available() and self.resource_quota.on_gpu:
            return len(tf.config.list_physical_devices("GPU"))
        return 1

    # pylint: disable=attribute-defined-outside-init
    def _setup(self) -> None:
        # setup a global session for model runner
        self._session = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(**self._config_proto)
        )
        self._model = load(self._tag, model_store=self._model_store)
        raw_predict_fn = getattr(self._model, self._predict_fn_name)
        self._predict_fn = functools.partial(raw_predict_fn, **self._partial_kwargs)

    def _run_batch(self, *args: "TFArgType", **kwargs: "TFArgType") -> "ext.NpNDArray":
        params = Params["TFArgType"](*args, **kwargs)

        with tf.device(self._device_id):

            def _mapping(item: "TFArgType") -> "tf.Tensor":
                if not LazyType["tf.Tensor"]("tf.Tensor").isinstance(item):
                    return tf.convert_to_tensor(item, dtype=tf.float32)
                else:
                    return item

            params = params.map(_mapping)

            tf.compat.v1.global_variables_initializer()
            res = self._predict_fn(*params.args, **params.kwargs)
            return t.cast("ext.NpNDArray", res.numpy())


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    predict_fn_name: str = "__call__",
    device_id: str = "CPU:0",
    name: t.Optional[str] = None,
    partial_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_TensorflowRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.tensorflow.load_runner` implements a Runner class that
    wrap around a Tensorflow model, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (:code:`str`, default to :code:`__call__`):
            Inference function to be used.
        partial_kwargs (:code:`Dict[str, Any]`, `optional`, default to :code:`None`):
            Dictionary of partial kwargs that can be shared across different model.
        device_id (:code:`str`, `optional`, default to the first CPU):
            Optional devices to put the given model on. Refers to `Logical Devices <https://www.tensorflow.org/api_docs/python/tf/config/list_logical_devices>`_ from TF documentation.
        resource_quota (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        batch_options (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.tensorflow` model

    Examples:

    .. tabs::

        .. code-tab:: tensorflow_v1

            import bentoml

            # load a runner from a given flag
            runner = bentoml.tensorflow.load_runner(tag)

            # load a runner on GPU:0
            runner = bentoml.tensorflow.load_runner(tag, resource_quota=dict(gpus=0), device_id="GPU:0")

        .. code-tab:: tensorflow_v2

            import bentoml

            # load a runner from a given flag
            runner = bentoml.tensorflow.load_runner(tag)

            # load a runner on GPU:0
            runner = bentoml.tensorflow.load_runner(tag, resource_quota=dict(gpus=0), device_id="GPU:0")

    """  # noqa
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _TensorflowRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        device_id=device_id,
        partial_kwargs=partial_kwargs,
        name=name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
