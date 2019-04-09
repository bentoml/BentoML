# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from bentoml.artifact import ArtifactSpec, ArtifactInstance


class TfKerasModelArtifact(ArtifactSpec):
    """
    Abstraction for saving/loading tensorflow keras model
    """

    def __init__(self, name, model_extension='.h5'):
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
            raise ImportError("tensorflow package is required to use TfKerasModelArtifact")

        model = load_model(self._model_file_path(path))
        return self.pack(model)


class _TfKerasModelArtifactInstance(ArtifactInstance):

    def __init__(self, spec, model):
        super(_TfKerasModelArtifactInstance, self).__init__(spec)

        try:
            from tensorflow.python.keras.engine import training
        except ImportError:
            raise ImportError("tensorflow package is required to use TfKerasModelArtifact")

        if not isinstance(model, training.Model):
            raise ValueError('Expected `model` argument to be a `Model` instance')

        self._model = model

    def save(self, dst):
        return self._model.save(self.spec._model_file_path(dst))

    def get(self):
        return self._model
