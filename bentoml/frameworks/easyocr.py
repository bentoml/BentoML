import os
import json

from bentoml.exceptions import InvalidArgument, MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv


class EasyOCRArtifact(BentoServiceArtifact):
    """
    Artifact class  for saving/loading EasyOCR models

    Args:
        name (str): Name for the artifact

    Raises:
        MissingDependencyError: easyocr>=1.3 package is required for EasyOCRArtifact
        InvalidArgument: invalid argument type, model being packed
            must be easyocr.easyocr.Reader

    Example usage:

    >>> import bentoml
    >>> from bentoml.frameworks.easyocr import EasyOCRArtifact
    >>> from bentoml.adapters import ImageInput
    >>>
    >>> @bentoml.env(pip_packages=["easyocr>=1.3.0"])
    >>> @bentoml.artifacts([EasyOCRArtifact("chinese_small")])
    >>> class EasyOCRService(bentoml.BentoService):
    >>>     @bentoml.api(input=ImageInput(), batch=False)
    >>>     def predict(self, image):
    >>>         reader = self.artifacts.chinese_small
    >>>         raw_results = reader.readtext(np.array(image))
    >>>         text_instances = [x[1] for x in raw_results]
    >>>         return {"text" : text_instances}
    >>>
    >>> import easyocr
    >>> service = EasyOCRService()
    >>>
    >>> lang_list = ['ch_sim', 'en']
    >>> recog_network = "zh_sim_g2"
    >>>
    >>> model = easyocr.Reader(lang_list=lang_list, download_enabled=True, \
    >>>                        recog_network=recog_network)
    >>> service.pack('chinese_small', model, lang_list=lang_list, \
    >>>              recog_network= recog_network)
    >>>
    >>> saved_path = service.save()

    """

    def __init__(self, name):
        super().__init__(name)

        self._model = None
        self._detect_model = None
        self._recog_network = None
        self._model_params = None
        self._gpu = None

    def set_dependencies(self, env: BentoServiceEnv):
        if env._infer_pip_packages:
            env.add_pip_packages(["easyocr>=1.3.0"])

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(
        self,
        easyocr_model,
        metadata=None,
        recog_network="english_g2",
        lang_list=None,
        detect_model="craft_mlt_25k",
        gpu=False,
    ):  # pylint:disable=arguments-differ
        try:
            import easyocr  # noqa # pylint: disable=unused-import

            assert easyocr.__version__ >= "1.3"
        except ImportError:
            raise MissingDependencyException(
                "easyocr>=1.3 package is required to use EasyOCRArtifact"
            )

        if not (type(easyocr_model) is easyocr.easyocr.Reader):
            raise InvalidArgument(
                "'easyocr_model' must be of type  easyocr.easyocr.Reader"
            )

        if not lang_list:
            lang_list = ['en']
        self._model = easyocr_model
        self._detect_model = detect_model
        self._recog_network = recog_network
        self._gpu = gpu
        self._model_params = {
            "lang_list": lang_list,
            "recog_network": recog_network,
            "gpu": gpu,
        }

        return self

    def load(self, path):
        try:
            import easyocr  # noqa # pylint: disable=unused-import

            assert easyocr.__version__ >= "1.3"
        except ImportError:
            raise MissingDependencyException(
                "easyocr>=1.3 package is required to use EasyOCRArtifact"
            )

        with open(os.path.join(path, f"{self._name}.json"), "r") as f:
            model_params = json.load(f)

        model = easyocr.Reader(
            model_storage_directory=path, download_enabled=False, **model_params
        )
        self._model = model

        return self.pack(model)

    def get(self):
        return self._model

    def save(self, dst):
        from shutil import copyfile

        src_folder = self._model.model_storage_directory

        detect_filename = f"{self._detect_model}.pth"

        if not os.path.exists(os.path.join(dst, detect_filename)):
            copyfile(
                os.path.join(src_folder, detect_filename),
                os.path.join(dst, detect_filename),
            )

        fname = self._recog_network + ".pth"
        copyfile(os.path.join(src_folder, fname), os.path.join(dst, fname))

        with open(os.path.join(dst, f"{self._name}.json"), "w") as f:
            json.dump(self._model_params, f)
