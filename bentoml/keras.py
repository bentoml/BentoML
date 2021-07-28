import os
import typing as t

import cloudpickle

from ._internal.models.base import (
    H5_EXTENSION,
    HDF5_EXTENSION,
    JSON_EXTENSION,
    MODEL_NAMESPACE,
    PICKLE_EXTENSION,
    Model,
)
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

# fmt: off
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise MissingDependencyException("tensorflow is required by KerasModel as backend runtime.")  # noqa
# fmt: on


class KerasModel(Model):
    """
    Model class for saving/loading :obj:`keras` models using Tensorflow backend.

    Args:
        model (`tf.keras.models.Model`):
            Keras model instance and its subclasses.
        store_as_json (`bool`, `optional`, default to `False`):
            Whether to store Keras model as JSON and weights
        custom_objects (`Dict[str, Any]`, `optional`, default to `None`):
            Dictionary of Keras custom objects for model
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`tensorflow` is required by KerasModel
        InvalidArgument:
            model being packed must be instance of :class:`tf.keras.models.Model`

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    _graph = tf.compat.v1.get_default_graph()
    # NOTES: sess should be user facing for V1 compatibility
    sess = tf.compat.v1.Session(graph=_graph)

    def __init__(
        self,
        model: "keras.models.Model",
        store_as_json: t.Optional[bool] = False,
        custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
        metadata: t.Optional[MetadataType] = None,
    ):
        super(KerasModel, self).__init__(model, metadata=metadata)

        self._store_as_json: t.Optional[bool] = store_as_json
        self._custom_objects: t.Optional[t.Dict[str, t.Any]] = custom_objects

    @staticmethod
    def __get_custom_obj_fpath(path: PathType) -> PathType:
        return os.path.join(path, f"{MODEL_NAMESPACE}_custom_objects{PICKLE_EXTENSION}")

    @staticmethod
    def __get_model_saved_fpath(path: PathType) -> PathType:
        return os.path.join(path, f"{MODEL_NAMESPACE}{H5_EXTENSION}")

    @staticmethod
    def __get_model_weight_fpath(path: PathType) -> PathType:
        return os.path.join(path, f"{MODEL_NAMESPACE}_weights{HDF5_EXTENSION}")

    @staticmethod
    def __get_model_json_fpath(path: PathType) -> PathType:
        return os.path.join(path, f"{MODEL_NAMESPACE}_json{JSON_EXTENSION}")

    @classmethod
    def load(cls, path: PathType) -> "keras.models.Model":
        default_custom_objects = None
        if os.path.isfile(cls.__get_custom_obj_fpath(path)):
            with open(cls.__get_custom_obj_fpath(path), "rb") as dco_file:
                default_custom_objects = cloudpickle.load(dco_file)

        with cls.sess.as_default():  # pylint: disable=not-context-manager
            if os.path.isfile(cls.__get_model_json_fpath(path)):
                # load keras model via json and weights since json file are in path
                with open(cls.__get_model_json_fpath(path), "r") as json_file:
                    model_json = json_file.read()
                obj = keras.models.model_from_json(
                    model_json, custom_objects=default_custom_objects
                )
                obj.load_weights(cls.__get_model_weight_fpath(path))
            else:
                # otherwise, load keras model via standard load_model
                obj = keras.models.load_model(
                    cls.__get_model_saved_fpath(path),
                    custom_objects=default_custom_objects,
                )
        if isinstance(obj, dict):
            model = obj["model"]
        else:
            model = obj

        return model

    def save(self, path: PathType) -> None:
        tf.compat.v1.keras.backend.get_session()

        # save custom_objects for model
        if self._custom_objects:
            with open(self.__get_custom_obj_fpath(path), "wb") as custom_object_file:
                cloudpickle.dump(self._custom_objects, custom_object_file)

        if self._store_as_json:
            # save keras model using json and weights if requested
            with open(self.__get_model_json_fpath(path), "w") as json_file:
                json_file.write(self._model.to_json())
            self._model.save_weights(self.__get_model_weight_fpath(path))
        else:
            # otherwise, save standard keras model
            self._model.save(self.__get_model_saved_fpath(path))
