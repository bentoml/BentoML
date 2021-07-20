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

import json
import os
import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import MissingDependencyException
from ._internal.types import MetadataType, PathType

try:
    import easyocr

    assert easyocr.__version__ >= "1.3"
except ImportError:
    raise MissingDependencyException("easyocr>=1.3 is required by EasyOCRModel")


class EasyOCRModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`easyocr` models

    Args:
        model (`easyocr.Reader`):
            easyocr model is of type :class:`easyocr.easyocr.Reader`
        language_list (`List[str]`, `optional`, default to `["en"]`):
            TODO:
        recog_network (`str`, `optional`, default to `english_g2`):
            TODO:
        detect_model (`str`, `optional`, default to `craft_mlt_25k`):
            TODO:
        gpu (`bool`, `optional`, default to `False`):
            Whether to enable GPU for easyocr
        metadata (`Dict[str, Any]`, or :obj:`~bentoml._internal.types.MetadataType`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`easyocr>=1.3` is required by EasyOCRModel
        InvalidArgument:
            model is not an instance of :class:`easyocr.easyocr.Reader`

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """  # noqa: E501

    def __init__(
        self,
        model: "easyocr.Reader",
        recog_network: t.Optional[str] = "english_g2",
        detect_model: t.Optional[str] = 'craft_mlt_25k',
        gpu: t.Optional[bool] = False,
        language_list: t.Optional[t.List[str]] = None,
        metadata: t.Optional[MetadataType] = None,
    ):
        super(EasyOCRModel, self).__init__(model, metadata=metadata)

        self._model: "easyocr.Reader" = model
        self._recog_network = recog_network
        self._detect_model = detect_model
        self._gpu = gpu

        if language_list is None:
            language_list = ['en']
        self._model_metadata = {
            "lang_list": language_list,
            "recog_network": recog_network,
            "gpu": gpu,
        }

    @classmethod
    def _json_file_path(cls, path: PathType) -> str:
        return cls.model_path(path, cls.JSON_FILE_EXTENSION)

    @classmethod
    def load(cls, path: PathType) -> "easyocr.Reader":
        with open(cls._json_file_path(path), "r") as f:
            model_params = json.load(f)

        return easyocr.Reader(
            model_storage_directory=path, download_enabled=False, **model_params
        )

    def save(self, path: PathType) -> None:
        import shutil

        src_folder: str = self._model.model_storage_directory

        detect_filename = f"{self._detect_model}{self.PTH_FILE_EXTENSION}"

        if not os.path.exists(os.path.join(path, detect_filename)):
            shutil.copyfile(
                os.path.join(src_folder, detect_filename),
                os.path.join(path, detect_filename),
            )

        fname: str = f"{self._recog_network}{self.PTH_FILE_EXTENSION}"
        shutil.copyfile(os.path.join(src_folder, fname), os.path.join(path, fname))

        with open(self._json_file_path(path), "w") as f:
            json.dump(self._model_metadata, f)
