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
import sys

from bentoml.artifact import BentoServiceArtifact, BentoServiceArtifactWrapper
from bentoml.artifact.fastai_model_artifact.fastai_model_for_serving import \
    FastaiModelForServing
from bentoml.artifact.fastai_model_artifact.utils import _import_torch_module
from bentoml.utils import Path


class FastaiModelArtifact(BentoServiceArtifact):
    """Saving and Loading FastAI Model

    Args:
        name (str): Name for the fastai model

    Raises:
        ImportError: Require fastai package to use Fast ai model artifact
        TypeError: invalid argument type, model being packed must be instance of
            fastai.basic_train.Learner
    """

    def __init__(self, name):
        if sys.version_info.major < 3 or sys.version_info.minor < 6:
            raise SystemError("fast ai requires python 3.6 version or higher")

        super(FastaiModelArtifact, self).__init__(name)
        self._file_name = name + '.pkl'

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self._file_name)

    def pack(self, model):  # pylint:disable=arguments-differ
        torch_module = _import_torch_module()
        if not isinstance(model, torch_module.nn.Module):
            raise TypeError(
                "Expect `model` argument to be `torch.nn.Module` instance"
            )
        serving_model = FastaiModelForServing(model)

        return _FastaiModelArtifactWrapper(self, serving_model)

    def load(self, path):
        torch_module = _import_torch_module()
        source = Path(os.path.join(path, self._file_name))
        state = torch_module.load(source)
        model = state.pop('model')
        return self.pack(model)


class _FastaiModelArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, model):
        super(_FastaiModelArtifactWrapper, self).__init__(spec)

        self._model = model

    def save(self, dst):
        self._model.export(file=self.spec._file_name)

        shutil.copyfile(
            os.path.join(self._model.path, self.spec._file_name),
            self.spec._model_file_path(dst),
        )

    def get(self):
        return self._model
