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


class XgboostModelArtifact(ArtifactSpec):
    """
    Abstraction for saving/loading objects with xgb.save_model and xgb.load_model
    """

    def __init__(self, name, model_extension='.model'):
        super(XgboostModelArtifact, self).__init__(name)
        self._model_extension = model_extension

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._model_extension)

    def pack(self, model):  # pylint:disable=arguments-differ
        return _XgboostModelArtifactInstance(self, model)

    def load(self, path):
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost package is required to use XgboostModelArtifact")
        bst = xgb.Booster()
        bst.load_model(self._model_file_path(path))

        return self.pack(bst)


class _XgboostModelArtifactInstance(ArtifactInstance):

    def __init__(self, spec, model):
        super(_XgboostModelArtifactInstance, self).__init__(spec)

        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost package is required to use XgboostModelArtifact")

        if not isinstance(model, xgb.core.Booster):
            raise ValueError('Expect `model` argument to be a `xgboost.core.Booster` instance')

        self._model = model

    def save(self, dst):
        return self._model.save_model(self.spec._model_file_path(dst))

    def get(self):
        return self._model
