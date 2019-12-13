# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import importlib

from bentoml.utils import cloudpickle
from bentoml.artifact import BentoServiceArtifact, BentoServiceArtifactWrapper
from bentoml.exceptions import (
    ArtifactLoadingException,
    MissingDependencyException,
    InvalidArgument,
)


class KerasModelArtifact(BentoServiceArtifact):
    """
    Abstraction for saving/loading Keras model

    Args:
        name (string): name of the artifact
        custom_objects (dict): dictionary of Keras custom objects for model
        store_as_json_and_weights (bool): flag allowing storage of the Keras
            model as JSON and weights

    Raises:
        MissingDependencyException: keras or tensorflow.keras package is required for
            KerasModelArtifact
        InvalidArgument:  invalid argument type, model being packed must be instance of
            keras.engine.network.Network, tf.keras.models.Model, or their aliases
    """

    def __init__(
        self,
        name,
        custom_objects=None,
        model_extension=".h5",
        store_as_json_and_weights=False,
    ):
        super(KerasModelArtifact, self).__init__(name)

        try:
            import tensorflow as tf
        except ImportError:
            raise MissingDependencyException(
                "Tensorflow package is required to use KerasModelArtifact. BentoML "
                "currently only support using Keras with Tensorflow backend."
            )

        self._model_extension = model_extension
        self._store_as_json_and_weights = store_as_json_and_weights

        # By default assume using tf.keras module
        self._keras_module_name = tf.keras.__name__

        self.custom_objects = custom_objects
        self.graph = None
        self.sess = None

    def _keras_module_name_path(self, base_path):
        # The name of the keras module used, can be 'keras' or 'tensorflow.keras'
        return os.path.join(base_path, self.name + '_keras_module_name.txt')

    def _custom_objects_path(self, base_path):
        return os.path.join(base_path, self.name + '_custom_objects.pkl')

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._model_extension)

    def _model_weights_path(self, base_path):
        return os.path.join(base_path, self.name + '_weights.hdf5')

    def _model_json_path(self, base_path):
        return os.path.join(base_path, self.name + '_json.json')

    def bind_keras_backend_session(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise MissingDependencyException(
                "Tensorflow package is required to use KerasModelArtifact. BentoML "
                "currently only support using Keras with Tensorflow backend."
            )

        self.sess = tf.compat.v1.keras.backend.get_session()
        self.graph = self.sess.graph

    def creat_session(self):
        try:
            import tensorflow as tf
        except ImportError:
            raise MissingDependencyException(
                "Tensorflow package is required to use KerasModelArtifact. BentoML "
                "currently only support using Keras with Tensorflow backend."
            )

        self.graph = tf.compat.v1.get_default_graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        tf.compat.v1.keras.backend.set_session(self.sess)

    def pack(self, data):  # pylint:disable=arguments-differ
        try:
            import tensorflow as tf
        except ImportError:
            raise MissingDependencyException(
                "Tensorflow package is required to use KerasModelArtifact. BentoML "
                "currently only support using Keras with Tensorflow backend."
            )

        if isinstance(data, dict):
            model = data['model']
            custom_objects = (
                data['custom_objects']
                if 'custom_objects' in data
                else self.custom_objects
            )
        else:
            model = data
            custom_objects = self.custom_objects

        if not isinstance(model, tf.keras.models.Model):
            error_msg = (
                "KerasModelArtifact#pack expects model argument to be type: "
                "keras.engine.network.Network, tf.keras.models.Model, or their "
                "aliases, instead got type: {}".format(type(model))
            )
            try:
                import keras

                if not isinstance(model, keras.engine.network.Network):
                    raise InvalidArgument(error_msg)
                else:
                    self._keras_module_name = keras.__name__
            except ImportError:
                raise InvalidArgument(error_msg)

        self.bind_keras_backend_session()
        model._make_predict_function()
        return _KerasModelArtifactWrapper(self, model, custom_objects)

    def load(self, path):
        if os.path.isfile(self._keras_module_name_path(path)):
            with open(self._keras_module_name_path(path), "rb") as text_file:
                keras_module_name = text_file.read().decode("utf-8")
                try:
                    keras_module = importlib.import_module(keras_module_name)
                except ImportError:
                    raise ArtifactLoadingException(
                        "Failed to import '{}' module when loading saved "
                        "KerasModelArtifact".format(keras_module_name)
                    )

        self.creat_session()

        if self.custom_objects is None and os.path.isfile(
            self._custom_objects_path(path)
        ):
            self.custom_objects = cloudpickle.load(
                open(self._custom_objects_path(path), 'rb')
            )

        with self.graph.as_default():
            with self.sess.as_default():
                # load keras model via json and weights if requested
                if self._store_as_json_and_weights:
                    with open(self._model_json_path(path), 'r') as json_file:
                        model_json = json_file.read()
                    model = keras_module.models.model_from_json(
                        model_json, custom_objects=self.custom_objects
                    )
                    model.load_weights(self._model_weights_path(path))
                # otherwise, load keras model via standard load_model
                else:
                    model = keras_module.models.load_model(
                        self._model_file_path(path), custom_objects=self.custom_objects
                    )
        return self.pack(model)


class _KerasModelArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, model, custom_objects):
        super(_KerasModelArtifactWrapper, self).__init__(spec)

        self.graph = spec.graph
        self.sess = spec.sess
        self._model = model
        self._custom_objects = custom_objects
        self._store_as_json_and_weights = spec._store_as_json_and_weights
        self._model_wrapper = _KerasModelWrapper(self._model, self.graph, self.sess)

    def save(self, dst):
        # save the keras module name to be used when loading
        with open(self.spec._keras_module_name_path(dst), "wb") as text_file:
            text_file.write(self.spec._keras_module_name.encode("utf-8"))

        # save custom_objects for model
        cloudpickle.dump(
            self._custom_objects, open(self.spec._custom_objects_path(dst), "wb")
        )

        # save keras model using json and weights if requested
        if self.spec._store_as_json_and_weights:
            with open(self.spec._model_json_path(dst), "w") as json_file:
                json_file.write(self._model.to_json())
            self._model.save_weights(self.spec._model_weights_path(dst))

        # otherwise, save standard keras model
        else:
            self._model.save(self.spec._model_file_path(dst))

    def get(self):
        return self._model_wrapper


class _KerasModelWrapper:
    def __init__(self, keras_model, graph, sess):
        self.keras_model = keras_model
        self.graph = graph
        self.sess = sess

    def predict(self, *args, **kwargs):
        with self.graph.as_default():
            with self.sess.as_default():
                return self.keras_model.predict(*args, **kwargs)

    def predict_classes(self, *args, **kwargs):
        with self.graph.as_default():
            with self.sess.as_default():
                return self.keras_model.predict_classes(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with self.graph.as_default():
            with self.sess.as_default():
                return object.__call__(self, *args, **kwargs)
