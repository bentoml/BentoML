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


class LightGBMModelArtifact(BentoServiceArtifact):
    """
    Abstraction for save/load object with LightGBM.

    Args:
        name (string): name of the artifact
        model_extension (string): Extension name for saved xgboost model

    Raises:
        MissingDependencyException: lightgbm package is required for using
            LightGBMModelArtifact
        InvalidArgument: invalid argument type, model being packed must be instance of
            lightgbm.Booster
    """

    def __init__(self, name, model_extension=".txt"):
        super(LightGBMModelArtifact, self).__init__(name)
        self.model_extension = model_extension

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self.model_extension)

    def pack(self, model):  # pylint:disable=arguments-differ
        return _LightGBMModelArtifactWrapper(self, model)

    def load(self, path):
        try:
            import lightgbm as lgb
        except ImportError:
            raise MissingDependencyException(
                "lightgbm package is required to use LightGBMModelArtifact"
            )
        bst = lgb.Booster(model_file=self._model_file_path(path))

        return self.pack(bst)


class _LightGBMModelArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, model):

        super(_LightGBMModelArtifactWrapper, self).__init__(spec)

        try:
            import lightgbm as lgb
        except ImportError:
            raise MissingDependencyException(
                "lightgbm package is required to use LightGBMModelArtifact"
            )

        if not isinstance(model, lgb.Booster):
            raise InvalidArgument(
                "Expect `model` argument to be a `lightgbm.Booster` instance"
            )

        self._model = model

    def save(self, dst):
        return self._model.save_model(self.spec._model_file_path(dst))

    def get(self):
        return self._model
