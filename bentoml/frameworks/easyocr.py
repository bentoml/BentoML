import os
import json

from bentoml.exceptions import MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv

class EasyOCRArtifact(BentoServiceArtifact):
    """
    Abstraction for saving/loading EasyOCR models

    Args:
        name (str): Name for the artifact

    Raises:
        MissingDependencyError: easyocr>=1.3 package is required for EasyOCRArtifact 

    Example Usage:

    #TODO 
    """

    def __init__(self, name):
        super(EasyOCRArtifact, self).__init__(name)

        self._model = None

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_packages(["easyocr>=1.3.0"])

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(self, easyocr_model, metadata=None, recog_network="english_g2", lang_list=['en'], detect_model="craft_mlt_25k"):  # pylint:disable=arguments-differ
        try:
            import easyocr  # noqa # pylint: disable=unused-import
            assert easyocr.__version__ >= "1.3"
        except ImportError:
            raise MissingDependencyException(
                "easyocr>=1.3 package is required to use EasyOCRArtifact"
            )
        self._model = easyocr_model
        self._detect_model = detect_model
        self._recog_network = recog_network

        self._model_params = {
            "lang_list" : lang_list,
            "recog_network" : recog_network
        }

        return self

    def load(self, path, gpu=False):
        try:
            import easyocr  # noqa # pylint: disable=unused-import
            assert easyocr.__version__ >= "1.3"
        except ImportError:
            raise MissingDependencyException(
                 "easyocr>=1.3 package is required to use EasyOCRArtifact"
            )

        with open(os.path.join(path, f"{self._name}.json"), "r") as f:
            model_params = json.load(f)

        model = easyocr.Reader(model_storage_directory=path, gpu=gpu, download_enabled=False, **model_params) 
        self._model = model
        
        return self.pack(model)

    def get(self):
        return self._model

    def save(self, dst):
        from shutil import copyfile
        src_folder = self._model.model_storage_directory

        detect_filename = f"{self._detect_model}.pth"
    
        if not os.path.exists(os.path.join(dst, detect_filename )):
            copyfile(os.path.join(src_folder, detect_filename), os.path.join(dst, detect_filename ))

        fname = self._recog_network+".pth"
        copyfile(os.path.join(src_folder, fname), os.path.join(dst,fname))

        with open(os.path.join(dst, f"{self._name}.json"), "w") as f:
            json.dump(self._model_params, f)
