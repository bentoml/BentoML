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

import os
import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import h2o

    if t.TYPE_CHECKING:
        import h2o.model
except ImportError:
    raise MissingDependencyException("h2o is required by H2oModel")


class H2oModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`h2o` models using :meth:`~h2o.saved_model` and :meth:`~h2o.load_model`

    Args:
        model (`h2o.model.model_base.ModelBase`):
            :obj:`ModelBase` for all h2o model instance
        metadata (`Dict[str, Any]`, or :obj:`~bentoml._internal.types.MetadataType`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`h2o` is required by H2oModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """  # noqa: E501

    def __init__(
        self,
        model: "h2o.model.model_base.ModelBase",
        metadata: t.Optional[MetadataType] = None,
    ):
        super(H2oModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "h2o.model.model_base.ModelBase":
        h2o.init()
        h2o.no_progress()
        get_path: str = cls.get_path(os.path.join(path, cls._MODEL_NAMESPACE))
        return h2o.load_model(get_path)

    def save(self, path: PathType) -> None:
        h2o.save_model(model=self._model, path=self.get_path(path), force=True)
