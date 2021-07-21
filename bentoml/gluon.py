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

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import mxnet
    from mxnet import gluon  # noqa # pylint disable=unused-import
except ImportError:
    raise MissingDependencyException("mxnet is required by GluonModel")


class GluonModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`mxnet.gluon` models

    Args:
        model (`mxnet.gluon.Block`):
            Every :obj:`mxnet.gluon` object is based on :obj:`mxnet.gluon.Block`
        metadata (`Dict[str, Any]`, or :obj:`~bentoml._internal.types.MetadataType`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`mxnet` is required by GluonModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """  # noqa: E501

    def __init__(
        self, model: "mxnet.gluon.Block", metadata: t.Optional[MetadataType] = None,
    ):
        super(GluonModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType) -> "mxnet.gluon.Block":
        json_path: str = cls.get_path(path, "-symbol.json")
        params_path: str = cls.get_path(path, "-0000.params")
        return gluon.nn.SymbolBlock.imports(json_path, ["data"], params_path)

    def save(self, path: PathType) -> None:
        self._model.export(self.get_path(path))
