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

from bentoml.artifact import ArtifactSpec, ArtifactInstance


class TfKerasModelArtifact(ArtifactSpec):
    """
    Abstraction for saving/loading tensorflow keras model
    """

    def __init__(self, name, model_extension=".h5"):
        super(TfKerasModelArtifact, self).__init__(name)
        self._model_extension = model_extension

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._model_extension)

    def pack(self, model):  # pylint:disable=arguments-differ
        return _TfKerasModelArtifactInstance(self, model)

    def load(self, path):
        try:
            from tensorflow.python.keras.models import load_model
        except ImportError:
            raise ImportError(
                "tensorflow package is required to use TfKerasModelArtifact"
            )

        model = load_model(self._model_file_path(path))
        model._make_predict_function()
        return self.pack(model)


class _TfKerasModelArtifactInstance(ArtifactInstance):
    def __init__(self, spec, model):
        super(_TfKerasModelArtifactInstance, self).__init__(spec)

        try:
            from tensorflow.python.keras.engine import training
        except ImportError:
            raise ImportError(
                "tensorflow package is required to use TfKerasModelArtifact"
            )

        if not isinstance(model, training.Model):
            raise ValueError("Expected `model` argument to be a `Model` instance")

        self._model = model

    def save(self, dst):
        return self._model.save(self.spec._model_file_path(dst))

    def get(self):
        return self._model
