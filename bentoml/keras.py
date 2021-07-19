# ==============================================================================
#     Copyright (c) 2021 Atalaya Tech. Inc
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

import importlib
import os
import typing as t

from ._internal.artifacts import BaseArtifact
from ._internal.exceptions import (
    ArtifactLoadingException,
    InvalidArgument,
    MissingDependencyException,
)
from ._internal.types import MetadataType, PathType
from ._internal.utils import cloudpickle

try:
    import keras
    import tensorflow as tf
except ImportError:
    raise MissingDependencyException(
        "tensorflow is required by KerasModel. Currently BentoML "
        "only supports using Keras with Tensorflow backend."
    )


MODULE_NAME_FILE_ENCODING = "utf-8"

KT = t.TypeVar("KT", bound=t.Union[tf.keras.models.Model, keras.engine.network.Network])


class KerasModel(BaseArtifact):
    """
    Model class for saving/loading :obj:`keras` models using Tensorflow backend.

    Args:
        model (`tf.keras.models.Model`, or `keras.engine.network.Network`):
            All of keras model instance and its subclasses.
        default_custom_objects (`Dict[str, Any]`, `optional`, default to `None`):
            dictionary of Keras custom objects for model
        store_as_json_and_weights (`bool`, `optional`, default to `False`):
            flag allowing storage of the Keras model as JSON and weights
        metadata (`Dict[str, Any]`, or :obj:`~bentoml._internal.types.MetadataType`, `optional`, default to `None`):
            Class metadata
        name (`str`, `optional`, default to `kerasmodel`):
            KerasModel instance name

    Raises:
        MissingDependencyException:
            either :obj:`keras` or :obj:`tensorflow` package is required for KerasModel
        InvalidArgument:
            model being packed must be instance of :class:`keras.engine.network.Network`,
            :class:`tf.keras.models.Model`, or its subclasses

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """  # noqa: E501

    def __init__(
        self,
        model: KT,
        default_custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
        store_as_json_and_weights: t.Optional[bool] = False,
        metadata: t.Optional[MetadataType] = None,
        name: t.Optional[str] = "kerasmodel",
    ):
        super(KerasModel, self).__init__(model, metadata=metadata, name=name)

        self._default_custom_objects = default_custom_objects
        self._store_as_json_and_weights = store_as_json_and_weights

        # By default assume using tf.keras module
        self._keras_module_name = tf.keras.__name__
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.graph = self.sess.graph

        self._custom_objects = None

    def _keras_module_name_path(self, path: PathType) -> PathType:
        # The name of the keras module used, can be 'keras' or 'tensorflow.keras'
        return self.model_path(path, "_keras_module_name.txt")

    def _custom_objects_path(self, path: PathType) -> PathType:
        return self.model_path(path, f"_custom_objects{self.PICKLE_FILE_EXTENSION}")

    def _model_file_path(self, path: PathType) -> PathType:
        return self.model_path(path, self.H5_FILE_EXTENSION)

    def _model_weights_path(self, path: PathType) -> PathType:
        return self.model_path(path, f"_weights{self.HDF5_FILE_EXTENSION}")

    def _model_json_path(self, path: PathType) -> PathType:
        return self.model_path(path, f"_json{self.JSON_FILE_EXTENSION}")

    def create_session(self):
        self.graph = tf.compat.v1.get_default_graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        tf.compat.v1.keras.backend.set_session(self.sess)

    def pack(self, data):
        if isinstance(data, dict):
            model = data["model"]
            custom_objects = (
                data["custom_objects"]
                if "custom_objects" in data
                else self._default_custom_objects
            )
        else:
            model = data
            custom_objects = self._default_custom_objects

        if not isinstance(model, tf.keras.models.Model):
            error_msg = rf"""\
                KerasModel#pack expects model argument to be type:
                keras.engine.network.Network, tf.keras.models.Model,
                or its subclasses, instead got type: {type(model)}
            """

            try:
                import keras
                if not isinstance(model, keras.engine.network.Network):
                    raise InvalidArgument(error_msg)
                else:
                    self._keras_module_name = keras.__name__
            except ImportError:
                raise InvalidArgument(error_msg)

        self._model = model
        self._custom_objects = custom_objects
        return self

    def load(self, path):
        if os.path.isfile(self._keras_module_name_path(path)):
            with open(self._keras_module_name_path(path), "rb") as text_file:
                keras_module_name = text_file.read().decode(MODULE_NAME_FILE_ENCODING)
                try:
                    keras_module = importlib.import_module(keras_module_name)
                except ImportError:
                    raise ArtifactLoadingException(
                        f"Failed to import '{keras_module_name}' module when"
                        "loading saved KerasModel"
                    )
        else:
            raise ArtifactLoadingException(
                "Failed to read keras model name from '{}' when loading saved "
                "KerasModel".format(self._keras_module_name_path(path))
            )

        if self._default_custom_objects is None and os.path.isfile(
            self._custom_objects_path(path)
        ):
            self._default_custom_objects = cloudpickle.load(
                open(self._custom_objects_path(path), "rb")
            )

        if self._store_as_json_and_weights:
            # load keras model via json and weights if store_as_json_and_weights=True
            self.create_session()
            with self.graph.as_default():
                with self.sess.as_default():
                    with open(self._model_json_path(path), "r") as json_file:
                        model_json = json_file.read()
                    model = keras_module.models.model_from_json(
                        model_json, custom_objects=self._default_custom_objects
                    )
                    model.load_weights(self._model_weights_path(path))
        else:
            # otherwise, load keras model via standard load_model
            model = keras_module.models.load_model(
                self._model_file_path(path),
                custom_objects=self._default_custom_objects,
            )
        return model

    def save(self, path: PathType) -> None:
        # save the keras module name to be used when loading
        with open(self._keras_module_name_path(path), "wb") as text_file:
            text_file.write(self._keras_module_name.encode(MODULE_NAME_FILE_ENCODING))

        # save custom_objects for model
        cloudpickle.dump(
            self._custom_objects, open(self._custom_objects_path(path), "wb")
        )

        if self._store_as_json_and_weights:
            # save keras model using json and weights if requested
            with open(self._model_json_path(path), "w") as json_file:
                json_file.write(self._model.to_json())
            self._model.save_weights(self._model_weights_path(path))
        else:
            # otherwise, save standard keras model
            self._model.save(self._model_file_path(path))