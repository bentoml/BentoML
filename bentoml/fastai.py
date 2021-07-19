# ==============================================================================
#     Copyright (c) 2021 Atalaya Tech. Inc
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

import typing as t

from ._internal.artifacts import BaseArtifact
from ._internal.exceptions import MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import fastai
    import fastai.basics  # noqa

    # fastai v2
    import fastai.learner
except ImportError:
    raise MissingDependencyException("fastai v2 is required by FastaiModel")


class FastaiModel(BaseArtifact):
    """
    Model class for saving/loading :obj:`fastai` model

    Args:
        model (`fastai.learner.Learner`):
            Learner model from fastai
        metadata (`Dict[str, Any]`, or :obj:`~bentoml._internal.types.MetadataType`, `optional`, default to `None`):
            Class metadata
        name (`str`, `optional`, default to `fastaimodel`):
            Name for the fastai v2 model

    Raises:
        MissingDependencyException:
            :obj:`fastai` is required by FastaiModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """  # noqa: E501

    def __init__(
        self,
        model: "fastai.learner.Learner",
        metadata: t.Optional[MetadataType] = None,
        name: t.Optional[str] = "fastaimodel",
    ):
        super(FastaiModel, self).__init__(model, metadata=metadata, name=name)

    @classmethod
    def load(cls, path: PathType) -> "fastai.basics.Learner":
        model = fastai.basics.load_learner(
            str(cls.get_path(path, cls.PICKLE_FILE_EXTENSION))
        )
        return model

    def save(self, path: PathType) -> None:
        fname: str = self.model_path(path, self.PICKLE_FILE_EXTENSION)
        self._model.export(fname=fname)
