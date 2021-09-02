import json
import os
import shutil
import typing as t

import bentoml._internal.constants as const

from ._internal.models.base import JSON_EXTENSION, MODEL_NAMESPACE, PTH_EXTENSION, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader
from .exceptions import BentoMLException

_exc = const.IMPORT_ERROR_MSG.format(
    fwr="easyocr",
    module=__name__,
    inst="`pip install easyocr`",
)

if t.TYPE_CHECKING:  # pylint: disable=unused-import # pragma: no cover
    import easyocr
else:
    easyocr = LazyLoader("easyocr", globals(), "easyocr", exc_msg=_exc)


class EasyOCRModel(Model):
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

    One then can define :code:`bento.py`::

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
        assert easyocr.__version__ >= "1.3", BentoMLException(
            "Only easyocr>=1.3 is supported by BentoML"
        )
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

    @staticmethod
    def __get_json_fpath(path: PathType) -> str:
        return os.path.join(path, f"{MODEL_NAMESPACE}{JSON_EXTENSION}")

    @classmethod
    def load(cls, path: PathType) -> "easyocr.Reader":
        with open(cls.__get_json_fpath(path), "r") as f:
            model_params = json.load(f)

        return easyocr.Reader(
            model_storage_directory=path, download_enabled=False, **model_params
        )

    def save(self, path: PathType) -> None:

        src_folder: str = self._model.model_storage_directory

        detect_filename: str = f"{self._detect_model}{PTH_EXTENSION}"

        if not os.path.exists(os.path.join(path, detect_filename)):
            shutil.copyfile(
                os.path.join(src_folder, detect_filename),
                os.path.join(path, detect_filename),
            )

        fname: str = f"{self._recog_network}{PTH_EXTENSION}"
        shutil.copyfile(os.path.join(src_folder, fname), os.path.join(path, fname))

        with open(self.__get_json_fpath(path), "w") as f:
            json.dump(self._model_metadata, f)
