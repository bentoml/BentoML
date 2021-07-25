import json
import os
import typing as t

from ._internal.artifacts import ModelArtifact
from ._internal.types import MetadataType, PathType
from .exceptions import MissingDependencyException

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
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
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
    """

    def __init__(
        self,
        model: "easyocr.Reader",
        recog_network: t.Optional[str] = "english_g2",
        detect_model: t.Optional[str] = "craft_mlt_25k",
        gpu: t.Optional[bool] = False,
        language_list: t.Optional[t.List[str]] = None,
        metadata: t.Optional[MetadataType] = None,
    ):
        if not language_list:
            language_list = ["en"]
        super(EasyOCRModel, self).__init__(model, metadata=metadata)
        self._model: "easyocr.Reader" = model
        self._recog_network = recog_network
        self._detect_model = detect_model
        self._gpu = gpu

        self._model_metadata = {
            "lang_list": language_list,
            "recog_network": recog_network,
            "gpu": gpu,
        }

    @classmethod
    def __get_json_fpath(cls, path: PathType) -> str:
        return str(cls.get_path(path, cls.JSON_EXTENSION))

    @classmethod
    def load(cls, path: PathType) -> "easyocr.Reader":
        with open(cls.__get_json_fpath(path), "r") as f:
            model_params = json.load(f)

        return easyocr.Reader(
            model_storage_directory=path, download_enabled=False, **model_params
        )

    def save(self, path: PathType) -> None:
        import shutil

        src_folder: str = self._model.model_storage_directory

        detect_filename: str = f"{self._detect_model}{self.PTH_EXTENSION}"

        if not os.path.exists(os.path.join(path, detect_filename)):
            shutil.copyfile(
                os.path.join(src_folder, detect_filename),
                os.path.join(path, detect_filename),
            )

        fname: str = f"{self._recog_network}{self.PTH_EXTENSION}"
        shutil.copyfile(os.path.join(src_folder, fname), os.path.join(path, fname))

        with open(self.__get_json_fpath(path), "w") as f:
            json.dump(self._model_metadata, f)
