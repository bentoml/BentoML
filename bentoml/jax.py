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


class FlaxModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`flax` models

    Args:
        model (``):
            TODO:
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`jax` and :obj:`flax` are required by FlaxModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    try:
        import jax
    except ImportError:
        raise MissingDependencyException("jax is required by FlaxModel")

    def __init__(
        self, model, metadata: t.Optional[MetadataType] = None,
    ):
        super(FlaxModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType):
        try:
            import flax
        except ImportError:
            raise MissingDependencyException("flax is required by FlaxModel")
        print(flax.__version__)

    def save(self, path: PathType) -> None:
        try:
            import flax
        except ImportError:
            raise MissingDependencyException("flax is required by FlaxModel")
        print(flax.__version__)


class TraxModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`trax` models

    Args:
        model (``):
            TODO:
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`jax` and :obj:`trax` are required by TraxModel

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """

    try:
        import jax
    except ImportError:
        raise MissingDependencyException("jax is required by TraxModel")

    def __init__(
        self, model, metadata: t.Optional[MetadataType] = None,
    ):
        super(TraxModel, self).__init__(model, metadata=metadata)

    @classmethod
    def load(cls, path: PathType):
        try:
            import trax
        except ImportError:
            raise MissingDependencyException("trax is required by FlaxModel")
        print(trax.models.GRULM)

    def save(self, path: PathType) -> None:
        try:
            import trax
        except ImportError:
            raise MissingDependencyException("trax is required by FlaxModel")
        print(trax.models.GRULM)
