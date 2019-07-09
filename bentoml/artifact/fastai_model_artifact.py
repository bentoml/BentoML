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
import sys
import shutil

from bentoml.artifact import ArtifactSpec, ArtifactInstance


class FastaiModelArtifact(ArtifactSpec):
    """Saving and Loading FastAI Model

    Args:
        name (str): Name for the fastai model

    Raises:
        ImportError: Require fastai package to use Fast ai model artifact
        ValueError: Model is not instance of fast ai model
    """

    def __init__(self, name):
        super(FastaiModelArtifact, self).__init__(name)
        self._file_name = name + '.pkl'

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self._file_name)

    def pack(self, model):  # pylint:disable=arguments-differ
        return _FastaiModelArtifactInstance(self, model)

    def load(self, path):
        try:
            # We need matplotlib and torch for fastai.  Make sure we install those.
            from fastai.basic_train import load_learner
        except ImportError:
            raise ImportError('fastai package is required to use FastaiModelArtifact')

        model = load_learner(path, self._file_name)
        return self.pack(model)


class _FastaiModelArtifactInstance(ArtifactInstance):
    def __init__(self, spec, model):
        super(_FastaiModelArtifactInstance, self).__init__(spec)
        if sys.version_info.major < 3 or sys.version_info.minor < 6:
            raise SystemError("fast ai requires python 3.6 version or higher")

        try:
            import fastai
        except ImportError:
            raise ImportError("fastai package is required to use FastaiModelArtifact")

        if not isinstance(model, fastai.basic_train.Learner):
            raise ValueError(
                "Expect `model` argument to be `fastai.basic_train.Learner` instance"
            )

        self._model = model

    def save(self, dst):
        self._model.export(file=self.spec._file_name)

        shutil.copyfile(
            os.path.join(self._model.path, self.spec._file_name),
            self.spec._model_file_path(dst),
        )
        return

    def get(self):
        return self._model
