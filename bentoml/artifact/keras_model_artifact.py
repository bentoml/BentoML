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

from bentoml.utils import cloudpickle
from bentoml.artifact import ArtifactSpec, ArtifactInstance

try:
    import tensorflow as tf
    from tensorflow.python import keras
except ImportError:
    tf = None
    keras = None


class KerasModelArtifact(ArtifactSpec):
    """
    Abstraction for saving/loading Keras model
    """

    def __init__(self, name, custom_objects=None, model_extension=".h5"):
        super(KerasModelArtifact, self).__init__(name)
        self._model_extension = model_extension

        self.custom_objects = custom_objects
        self.graph = None
        self.sess = None

    def _custom_objects_path(self, base_path):
        return os.path.join(base_path, self.name + '_custom_objects.pkl')

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._model_extension)

    def bind_keras_backend_session(self):
        if tf is None:
            raise ImportError(
                "Tensorflow package is required to use KerasModelArtifact."
            )

        self.sess = tf.compat.v1.keras.backend.get_session()
        self.graph = self.sess.graph

    def creat_session(self):
        if tf is None:
            raise ImportError(
                "Tensorflow package is required to use KerasModelArtifact."
            )

        self.graph = tf.compat.v1.get_default_graph()
        self.sess = tf.Session(graph=self.graph)
        tf.keras.backend.set_session(self.sess)

    def pack(self, data):  # pylint:disable=arguments-differ
        if tf is None:
            raise ImportError(
                "Tensorflow package is required to use KerasModelArtifact."
            )

        if isinstance(data, keras.engine.training.Model):
            model = data
            custom_objects = self.custom_objects
        elif (
            isinstance(data, dict)
            and 'model' in data
            and isinstance(data['model'], keras.engine.training.Model)
        ):
            model = data['model']
            custom_objects = (
                data['custom_objects']
                if 'custom_objects' in data
                else self.custom_objects
            )
        else:
            raise ValueError(
                "KerasModelArtifact#pack expects type: keras.engine.training.Model"
            )

        self.bind_keras_backend_session()
        model._make_predict_function()
        return _TfKerasModelArtifactInstance(self, model, custom_objects)

    def load(self, path):
        if tf is None:
            raise ImportError(
                "Tensorflow package is required to use KerasModelArtifact."
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
                model = keras.models.load_model(
                    self._model_file_path(path), custom_objects=self.custom_objects
                )
        return self.pack(model)


class _TfKerasModelArtifactInstance(ArtifactInstance):
    def __init__(self, spec, model, custom_objects):
        super(_TfKerasModelArtifactInstance, self).__init__(spec)

        if tf is None:
            raise ImportError(
                "Tensorflow package is required to use KerasModelArtifact."
            )

        if not isinstance(model, keras.engine.training.Model):
            raise ValueError(
                "Expected `model` argument to be a "
                "`keras.engine.training.Model` instance"
            )

        self.graph = spec.graph
        self.sess = spec.sess
        self._model = model
        self._custom_objects = custom_objects
        self._model_wrapper = _TfKerasModelWrapper(self._model, self.graph, self.sess)

    def save(self, dst):
        # save custom_objects for model
        cloudpickle.dump(
            self._custom_objects, open(self.spec._custom_objects_path(dst), "wb")
        )

        # save keras model
        self._model.save(self.spec._model_file_path(dst))

    def get(self):
        return self._model_wrapper


class _TfKerasModelWrapper:
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
