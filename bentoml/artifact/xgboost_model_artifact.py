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

from bentoml.artifact import BentoServiceArtifact, BentoServiceArtifactWrapper
from bentoml.exceptions import MissingDependencyException, InvalidArgument


class XgboostModelArtifact(BentoServiceArtifact):
    """Abstraction for save/load object with Xgboost.

    Args:
        name (string): name of the artifact
        model_extension (string): Extension name for saved xgboost model

    Raises:
        ImportError: xgboost package is required for using XgboostModelArtifact
        TypeError: invalid argument type, model being packed must be instance of
            xgboost.core.Booster
    """

    def __init__(self, name, model_extension=".model"):
        super(XgboostModelArtifact, self).__init__(name)
        self._model_extension = model_extension

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._model_extension)

    def pack(self, model):  # pylint:disable=arguments-differ
        return _XgboostModelArtifactWrapper(self, model)

    def load(self, path):
        try:
            import xgboost as xgb
        except ImportError:
            raise MissingDependencyException(
                "xgboost package is required to use XgboostModelArtifact"
            )
        bst = xgb.Booster()
        bst.load_model(self._model_file_path(path))

        return self.pack(bst)


class _XgboostModelArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, model):
        super(_XgboostModelArtifactWrapper, self).__init__(spec)

        try:
            import xgboost as xgb
        except ImportError:
            raise MissingDependencyException(
                "xgboost package is required to use XgboostModelArtifact"
            )

        if not isinstance(model, xgb.core.Booster):
            raise InvalidArgument(
                "Expect `model` argument to be a `xgboost.core.Booster` instance"
            )

        self._model = model

    def save(self, dst):
        return self._model.save_model(self.spec._model_file_path(dst))

    def get(self):
        return self._model
