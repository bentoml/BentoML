# type: ignore[reportMissingTypeStubs]
import typing as t
import functools
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import cloudpickle
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import H5_EXT
from ..models import PKL_EXT
from ..models import HDF5_EXT
from ..models import JSON_EXT
from ..models import SAVE_NAMESPACE
from ..utils.tensorflow import get_tf_version
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow.python.framework.ops import Graph
    from tensorflow.python.client.session import BaseSession

    from ..models import ModelStore

try:
    import tensorflow as tf

    # TODO: separation of Keras and Tensorflow
    # https://twitter.com/fchollet/status/1404967230048149506?lang=en
    import tensorflow.keras as keras
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
        `tensorflow` is required to use `bentoml.keras`, since
        we will use Tensorflow as Keras backend.\n
        Instruction: Refers to https://www.tensorflow.org/install for
        more information of your use case.
        """
    )

MODULE_NAME = "bentoml.keras"

_tf_version = get_tf_version()
TF2 = _tf_version.startswith("2")

if TF2:
    from .tensorflow_v2 import _TensorflowRunner  # type: ignore[reportPrivateUsage]
else:
    from .tensorflow_v1 import _TensorflowRunner  # type: ignore[reportPrivateUsage]

# Global instance of tf.Session()
_graph: "Graph" = tf.compat.v1.get_default_graph()
_sess: "BaseSession" = tf.compat.v1.Session(graph=_graph)


_CUSTOM_OBJ_FNAME = f"{SAVE_NAMESPACE}_custom_objects{PKL_EXT}"
_SAVED_MODEL_FNAME = f"{SAVE_NAMESPACE}{H5_EXT}"
_MODEL_WEIGHT_FNAME = f"{SAVE_NAMESPACE}_weights{HDF5_EXT}"
_MODEL_JSON_FNAME = f"{SAVE_NAMESPACE}_json{JSON_EXT}"


def get_session() -> "BaseSession":
    """
    Return TF1 sessions for :mod:`bentoml.keras`.

    .. warning::

       This function is served for the purposes of using Tensorflow V1.

    Example:

    .. tabs::

        .. code-tab:: keras_v1

            session = bentoml.keras.get_session()
            # Initialize variables in the graph/model
            session.run(tf.global_variables_initializer())
            with session.as_default():
                loaded = bentoml.keras.load(tag)

    """  # noqa: LN001
    return _sess


@inject
def load(
    tag: Tag,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "keras.Model":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`keras.Model`: an instance of users :obj:`keras.Model` from BentoML modelstore.

    Examples:

    .. tabs::

        .. code-tab:: keras_v1

            import bentoml
            import tensorflow as tf

            # retrieve session that save keras model with `bentoml.keras.get_session()`:
            session = bentoml.keras.get_session()
            session.run(tf.global_variables_initializer())
            with session.as_default():
                # `load` the model back in memory:
                loaded = bentoml.keras.load("keras_model")

        .. code-tab:: keras_v2

            import bentoml

            # `load` the model back in memory:
            loaded = bentoml.keras.load("keras_model")

    """  # noqa

    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )

    default_custom_objects = None
    if model.info.options["custom_objects"]:
        assert Path(model.path_of(_CUSTOM_OBJ_FNAME)).is_file()
        with Path(model.path_of(_CUSTOM_OBJ_FNAME)).open("rb") as dcof:
            default_custom_objects = cloudpickle.load(dcof)

    with get_session().as_default():
        if model.info.options["store_as_json"]:
            assert Path(model.path_of(_MODEL_JSON_FNAME)).is_file()
            with Path(model.path_of(_MODEL_JSON_FNAME)).open("r") as jsonf:
                model_json = jsonf.read()
            model = keras.models.model_from_json(
                model_json, custom_objects=default_custom_objects
            )
        else:
            model = keras.models.load_model(
                model.path_of(_SAVED_MODEL_FNAME),
                custom_objects=default_custom_objects,
            )
        try:
            # if model is a dictionary
            return model["model"]
        except TypeError:
            return model


@inject
def save(
    name: str,
    model: "keras.Model",
    *,
    store_as_json: t.Optional[bool] = False,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`tensorflow.keras.Model`):
            Instance of the Keras model to be saved to BentoML modelstore.
        store_as_json (`bool`, `optional`, default to :code:`False`):
            Whether to store Keras model as JSON and weights.
        custom_objects (:code:`Dict[str, Any]`, `optional`, default to :code:`None`):
            Dictionary of Keras custom objects, if specified.
        metadata (:code:`Dict[str, Any]`, `optional`, default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. tabs::

        .. code-tab:: keras_v1

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
            tag = bentoml.keras.save("keras_model", model, store_as_json=True)

            # `save` a given model with custom objects definition:
            custom_objects = {
                "CustomLayer": CustomLayer,
                "custom_activation": custom_activation,
            },
            custom_tag = bentoml.keras.save("custom_obj_keras", custom_objects=custom_objects)

        .. code-tab:: keras_v2

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
            tag = bentoml.keras.save("keras_model", model, store_as_json=True)

            # `save` a given model with custom objects definition:
            custom_objects = {
                "CustomLayer": CustomLayer,
                "custom_activation": custom_activation,
            },
            custom_tag = bentoml.keras.save("custom_obj_keras", custom_objects=custom_objects)

    """  # noqa
    tf.compat.v1.keras.backend.get_session()
    context: t.Dict[str, t.Any] = {
        "framework_name": "keras",
        "pip_dependencies": [f"tensorflow=={_tf_version}"],
    }
    options = {
        "store_as_json": store_as_json,
        "custom_objects": True if custom_objects is not None else False,
    }
    _model = Model.create(
        name,
        module=MODULE_NAME,
        options=options,
        context=context,
        metadata=metadata,
    )

    if custom_objects is not None:
        with Path(_model.path_of(_CUSTOM_OBJ_FNAME)).open("wb") as cof:
            cloudpickle.dump(custom_objects, cof)
    if store_as_json:
        with Path(_model.path_of(_MODEL_JSON_FNAME)).open("w") as jf:
            jf.write(model.to_json())
        model.save_weights(_model.path_of(_MODEL_WEIGHT_FNAME))
    else:
        model.save(_model.path_of(_SAVED_MODEL_FNAME))

    _model.save(model_store)

    return _model.tag


class _KerasRunner(_TensorflowRunner):
    @inject
    def __init__(
        self,
        tag: Tag,
        predict_fn_name: str,
        device_id: str,
        predict_kwargs: t.Optional[t.Dict[str, t.Any]],
        name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(
            tag=tag,
            predict_fn_name=predict_fn_name,
            device_id=device_id,
            name=name,
            partial_kwargs=predict_kwargs,
            resource_quota=resource_quota,
            batch_options=batch_options,
            model_store=model_store,
        )
        self._session = get_session()

    @property
    def session(self) -> "BaseSession":
        return self._session

    # pylint: disable=attribute-defined-outside-init
    def _setup(self) -> None:
        self._session.config = self._config_proto
        self._model = load(self._tag, model_store=self._model_store)
        raw_predict_fn = getattr(self._model, self._predict_fn_name)
        self._predict_fn = functools.partial(raw_predict_fn, **self._partial_kwargs)

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[override]
        self,
        input_data: t.Union[
            t.List[t.Union[int, float]], "np.ndarray[t.Any, np.dtype[t.Any]]", tf.Tensor
        ],
    ) -> t.Union["np.ndarray[t.Any, np.dtype[t.Any]]", tf.Tensor]:
        if not isinstance(input_data, (np.ndarray, tf.Tensor)):
            input_data = np.array(input_data)
        with tf.device(self._device_id):
            if TF2:
                tf.compat.v1.global_variables_initializer()
            else:
                self._session.run(tf.compat.v1.global_variables_initializer())
                with get_session().as_default():
                    self._predict_fn(input_data)
            return self._predict_fn(input_data)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    predict_fn_name: str = "predict",
    device_id: str = "CPU:0",
    predict_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    name: t.Optional[str] = None,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_KerasRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.tensorflow.load_runner` implements a Runner class that
    wrap around a Keras model, with Tensorflow as backend, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (:code:`str`, `optional`, default to :code:`predict`):
            Inference function to be used.
        predict_kwargs (:code:`Dict[str, Any]`, `optional`, default to :code:`None`):
            Dictionary of `predict()` kwargs that can be shared across different model.
        device_id (:code:`str`, `optional`, default to the first CPU):
            Optional devices to put the given model on. Refers to `Logical Devices <https://www.tensorflow.org/api_docs/python/tf/config/list_logical_devices>`_ from TF documentation.
        resource_quota (:code:`Dict[str, Any]`, `optional`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        batch_options (:code:`Dict[str, Any]`, `optional`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.keras` model

    Examples:

    .. tabs::

        .. code-tab:: keras_v1

            import bentoml

            # load tag to a BentoML runner:
            runner = bentoml.keras.load_runner(tag)
            with bentoml.keras.get_session().as_default():
                runner.run_batch([1,2,3])

        .. code-tab:: keras_v2

            import bentoml

            # load tag to a BentoML runner:
            runner = bentoml.keras.load_runner(tag)
            runner.run_batch([1,2,3])

    """  # noqa: LN001
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _KerasRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        device_id=device_id,
        name=name,
        predict_kwargs=predict_kwargs,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
