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

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..types import LazyType
from ..runner.utils import Params
from ..utils.tensorflow import get_tf_version
from ..utils.tensorflow import is_gpu_available
from ..utils.tensorflow import hook_loaded_model
from .common.model_runner import BaseModelRunner
from ..configuration.containers import BentoMLContainer

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

MODULE_NAME = "bentoml.tensorflow_v1"


def _clean_name(name: str) -> str:  # pragma: no cover
    if name.startswith(("http://", "https://")):
        name = name.split("/", maxsplit=3)[-1]
    else:
        name = name.split("/")[-1]
    return re.sub(r"\W|^(?=\d)-", "_", name)


AUTOTRACKABLE_CALLABLE_WARNING: str = """\
Importing SavedModels from TensorFlow 1.x. `outputs = imported(inputs)`
 will not be supported by BentoML due to `tensorflow` API.\n
See https://www.tensorflow.org/api_docs/python/tf/saved_model/load for
 more details.
"""


@inject
def load(
    bento_tag: t.Union[str, Tag],
    tags: t.Optional[t.List[str]] = None,
    options: t.Optional["tf_ext.SaveOptions"] = None,
    load_as_hub_module: t.Optional[bool] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
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
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`SavedModel`: an instance of :obj:`SavedModel` format from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        # load a model back into memory
        model = bentoml.tensorflow_v1.load("my_tensorflow_model")
    """
    model = model_store.get(bento_tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {bento_tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    if model.info.context["import_from_tfhub"]:
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

        module_path = model.path_of(model.info.options["local_path"])
        if load_as_hub_module:
            return (
                hub.Module(module_path)
                if get_tf_version().startswith("1")
                else hub.KerasLayer(module_path)
            )
        # In case users want to load as a SavedModel file object.
        # https://github.com/tensorflow/hub/blob/master/tensorflow_hub/module_v2.py#L93
        is_hub_module_v1: bool = tf.io.gfile.exists(
            native_module.get_module_proto_path(module_path)
        )
        if tags is None and is_hub_module_v1:
            tags = []
        if options is not None:
            if not LazyType(
                "tensorflow.python.saved_model.save_options.SaveOptions"
            ).isinstance(options):
                raise BentoMLException(
                    f"`options` has to be of type `tf.saved_model.SaveOptions`, got {type(options)} instead."
                )
            if not hasattr(getattr(tf, "saved_model", None), "LoadOptions"):
                raise NotImplementedError(
                    "options are not supported for TF < 2.3.x,"
                    f" Current version: {get_tf_version()}"
                )
            tf_model: "tf_ext.AutoTrackable" = tf.saved_model.load_v2(  # type: ignore
                module_path, tags=tags, options=options
            )
        else:
            tf_model: "tf_ext.AutoTrackable" = tf.saved_model.load_v2(  # type: ignore
                module_path, tags=tags
            )
        tf_model._is_hub_module_v1 = (
            is_hub_module_v1  # pylint: disable=protected-access # noqa
        )
        return tf_model
    else:
        tf_model: "tf_ext.AutoTrackable" = tf.saved_model.load_v2(model.path)  # type: ignore
        if LazyType["tf_ext.AutoTrackable"](
            "tensorflow.python.training.tracking.tracking.AutoTrackable"
        ).isinstance(tf_model) and not hasattr(tf_model, "__call__"):
            logger.warning(AUTOTRACKABLE_CALLABLE_WARNING)
        return hook_loaded_model(tf_model, MODULE_NAME)


def import_from_tfhub(
    identifier: t.Union[str, "HubModule"],
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
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`~bentoml.Tag` object that can be used to retrieve the model with :func:`bentoml.tensorflow.load`:

    Example for importing a model from Tensorflow Hub:

    .. code-block:: python

        import tensorflow_text as text # noqa # pylint: disable
        import bentoml

        tag = bentoml.tensorflow_v1.import_from_tfhub("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

        # load model back with `load`:
        model = bentoml.tensorflow_v1.load(tag, load_as_hub_module=True)

    Example for importing a custom Tensorflow Hub model:

    .. code-block:: python

        import tensorflow as tf
        import tensorflow_hub as hub
        import bentoml

        # Simple toy Tensorflow Hub model
        def _plus_one_model_tf1():
            def plus_one():
                x = tf.placeholder(dtype=tf.float32, name="x")
                y = x + 1
                hub.add_signature(inputs=x, outputs=y)

            spec = hub.create_module_spec(plus_one)
            with tf.get_default_graph().as_default():
                module = hub.Module(spec, trainable=True)
                return module

        model = _plus_one_model_tf1()

        # retrieve the given tag:
        tag = bentoml.tensorflow_v1.import_from_tfhub(model)

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
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
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
                with tf.Session(graph=tf.get_default_graph()) as sess:  # type: ignore
                    sess.run(tf.global_variables_initializer())  # type: ignore
                    identifier.export(_model.path, sess)  # type: ignore
            else:
                tf.saved_model.save(identifier, _model.path)
            _model.info.options = {
                "model": identifier.__class__.__name__,
                "local_path": ".",
            }

        return _model.tag


def save(
    name: str,
    model: t.Union["PathType", "tf_ext.KerasModel", "tf_ext.Module"],
    *,
    signatures: t.Optional["tf_ext.ConcreteFunction"] = None,
    options: t.Optional["tf_ext.SaveOptions"] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[tf.keras.Model, tf.Module, path-like objects]`):
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
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json

    Raises:
        ValueError: If :obj:`obj` is not trackable.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import tensorflow as tf
        import tempfile
        import bentoml

        location = ""

        # Function below builds model graph
        def cnn_model_fn():
            X = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="X")

            # dense layer
            inter1 = tf.layers.dense(inputs=X, units=1, activation=tf.nn.relu)
            p = tf.argmax(input=inter1, axis=1)

            # loss
            y = tf.placeholder(tf.float32, shape=[None, 1], name="y")
            loss = tf.losses.softmax_cross_entropy(y, inter1)

            # training operation
            train_op = tf.train.AdamOptimizer().minimize(loss)

            return {"p": p, "loss": loss, "train_op": train_op, "X": X, "y": y}

        cnn_model = cnn_model_fn()

        with tempfile.TemporaryDirectory() as temp_dir:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(cnn_model["p"], {cnn_model["X"]: test_data})

                inputs = {"X": cnn_model["X"]}
                outputs = {"prediction": cnn_model["p"]}

                tf.saved_model.simple_save(
                    sess, temp_dir, inputs=inputs, outputs=outputs
                )
                location = temp_dir

        # then save the given model to BentoML modelstore:
        tag = bentoml.tensorflow_v1.save("cnn_model", location)

    .. note::

       :code:`bentoml.tensorflow.save` API also support saving `RaggedTensor <https://www.tensorflow.org/guide/ragged_tensor>`_ model and Keras model. If you choose to save a Keras model
       with :code:`bentoml.tensorflow.save`, then the model will be saved under a :obj:`SavedModel` format instead of :obj:`.h5`.

    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "tensorflow",
        "pip_dependencies": [f"tensorflow=={get_tf_version()}"],
        "import_from_tfhub": False,
    }
    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        options=None,
        context=context,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
    ) as _model:
        if isinstance(model, (str, bytes, os.PathLike, pathlib.Path)):  # type: ignore[reportUnknownMemberType]
            assert os.path.isdir(model)
            copy_tree(str(model), _model.path)
        else:
            if options:
                logger.warning(
                    f"Parameter 'options: {str(options)}' is ignored when "
                    f"using tensorflow {get_tf_version()}"
                )
            tf.saved_model.save(model, _model.path, signatures=signatures)

        return _model.tag


class _TensorflowRunner(BaseModelRunner):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        predict_fn_name: str,
        device_id: str,
        partial_kwargs: t.Optional[t.Dict[str, t.Any]],
        name: t.Optional[str] = None,
    ):
        super().__init__(tag=tag, name=name)

        self._device_id = device_id
        self._predict_fn_name = predict_fn_name
        self._partial_kwargs: t.Dict[str, t.Any] = (
            partial_kwargs if partial_kwargs is not None else {}
        )

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
    def _num_threads(self) -> int:
        if is_gpu_available() and self.resource_quota.on_gpu:
            return 1
        return max(round(self.resource_quota.cpu), 1)

    @property
    def num_replica(self) -> int:
        if is_gpu_available() and self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    def _setup(self) -> None:
        # setup a global session for model runner
        self._configure(self._device_id)
        self._model = load(self._tag, model_store=self.model_store)
        raw_predict_fn: t.Callable[..., t.Any] = self._model.signatures[  # type: ignore
            "serving_default"
        ]
        self._predict_fn = functools.partial(raw_predict_fn, **self._partial_kwargs)

    def _run_batch(
        self, *args: "TFArgType", **kwargs: "TFArgType"
    ) -> "tf_ext.TensorLike":
        params = Params["TFArgType"](*args, **kwargs)

        with tf.device(self._device_id):

            def _mapping(item: "TFArgType") -> "tf_ext.TensorLike":
                if not LazyType["tf_ext.TensorLike"]("tf.Tensor").isinstance(item):
                    return t.cast("tf_ext.TensorLike", tf.convert_to_tensor(item))
                else:
                    return item

            params = params.map(_mapping)

            with tf.Session(
                graph=tf.get_default_graph(),
                config=tf.ConfigProto(**self._config_proto),
            ) as sess:
                sess.run(tf.global_variables_initializer())  # type: ignore
                res = self._predict_fn(*params.args, **params.kwargs)["prediction"]  # type: ignore
                return t.cast("tf_ext.TensorLike", res)


def load_runner(
    tag: t.Union[str, Tag],
    *,
    predict_fn_name: str = "__call__",
    device_id: str = "CPU:0",
    name: t.Optional[str] = None,
    partial_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
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

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.tensorflow` model

    Examples:

    .. code-block:: python

        import bentoml

        # load a runner from a given flag
        runner = bentoml.tensorflow_v1.load_runner(tag)

        # load a runner on GPU:0
        runner = bentoml.tensorflow_v1.load_runner(tag, resource_quota=dict(gpus=0), device_id="GPU:0")

    """
    return _TensorflowRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        device_id=device_id,
        partial_kwargs=partial_kwargs,
        name=name,
    )
