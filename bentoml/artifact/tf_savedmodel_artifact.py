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
import shutil
import logging

from bentoml.utils import pathlib
from bentoml.artifact import BentoServiceArtifact, BentoServiceArtifactWrapper


logger = logging.getLogger(__name__)


def _is_path_like(p):
    return isinstance(p, (str, bytes, pathlib.PurePath, os.PathLike))


def _load_tf_saved_model(path):
    try:
        import tensorflow as tf

        TF2 = tf.__version__.startswith('2')
    except ImportError:
        raise ImportError("Tensorflow package is required to use TfSavedModelArtifact")

    if TF2:
        return tf.saved_model.load(path)
    else:
        return tf.compat.v2.saved_model.load(path)


class TensorflowSavedModelArtifact(BentoServiceArtifact):
    """
    Abstraction for saving/loading Tensorflow model in tf.saved_model format

    Args:
        name (string): name of the artifact

    Raises:
        ImportError: tensorflow package is required for TensorflowSavedModelArtifact

    Example usage:

    >>> import tensorflow as tf
    >>> class Adder(tf.Module):
    >>>     @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    >>>     def add(self, x):
    >>>         return x + x + 1.
    >>> to_export = Adder()
    >>>
    >>> import bentoml
    >>> from bentoml.handlers import JsonHandler
    >>> from bentoml.artifact import TensorflowSavedModelArtifact
    >>>
    >>> @bentoml.env(pip_dependencies=["tensorflow"])
    >>> @bentoml.artifacts([TensorflowSavedModelArtifact('model')])
    >>> class TfModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(JsonHandler)
    >>>     def predict(self, json):
    >>>         input_data = json['input']
    >>>         prediction = self.artifacts.model.add(input_data)
    >>>         return prediction.numpy()
    >>>
    >>> svc = TfModelService()
    >>>
    >>> # Option 1: pack directly with Tensorflow trackable object
    >>> svc.pack('model', to_export)
    >>>
    >>> # Option 2: save to file path then pack
    >>> tf.saved_model.save(to_export, '/tmp/adder/1')
    >>> svc.pack('model', '/tmp/adder/1')
    """

    def _saved_model_path(self, base_path):
        return os.path.join(base_path, self.name + '_saved_model')

    def pack(
        self, obj, signatures=None, options=None
    ):  # pylint:disable=arguments-differ
        """

        Args:
            obj: Either a path(str/byte/os.PathLike) containing exported
                `tf.saved_model` files, or a Trackable object mapping to the `obj`
                parameter of `tf.saved_model.save`
            signatures:
            options:
        """

        if _is_path_like(obj):
            return _ExportedTensorflowSavedModelArtifactWrapper(self, obj)

        return _TensorflowSavedModelArtifactWrapper(self, obj, signatures, options)

    def load(self, path):
        saved_model_path = self._saved_model_path(path)
        loaded_model = _load_tf_saved_model(saved_model_path)
        return self.pack(loaded_model)


class _ExportedTensorflowSavedModelArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, path):
        super(_ExportedTensorflowSavedModelArtifactWrapper, self).__init__(spec)

        self.path = path
        self.model = None

    def save(self, dst):
        # Copy exported SavedModel model directory to BentoML saved artifact directory
        shutil.copytree(self.path, self.spec._saved_model_path(dst))

    def get(self):
        if not self.model:
            self.model = _load_tf_saved_model(self.path)

        return self.model


class _TensorflowSavedModelArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, obj, signatures=None, options=None):
        super(_TensorflowSavedModelArtifactWrapper, self).__init__(spec)

        self.obj = obj
        self.signatures = signatures
        self.options = options

    def save(self, dst):
        try:
            import tensorflow as tf

            TF2 = tf.__version__.startswith('2')
        except ImportError:
            raise ImportError(
                "Tensorflow package is required to use TfSavedModelArtifact."
            )

        if TF2:
            return tf.saved_model.save(
                self.obj,
                self.spec._saved_model_path(dst),
                signatures=self.signatures,
                options=self.options,
            )
        else:
            if self.options:
                logger.warning(
                    "Parameter 'options: %s' is ignored when using Tensorflow "
                    "version 1",
                    str(self.options),
                )

            return tf.saved_model.save(
                self.obj, self.spec._saved_model_path(dst), signatures=self.signatures
            )

    def get(self):
        return self.obj
