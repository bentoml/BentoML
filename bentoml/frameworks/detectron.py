import os
from bentoml.exceptions import MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv


class DetectronModelArtifact(BentoServiceArtifact):

    def __init__(self, name):
        super(DetectronModelArtifact, self).__init__(name)
        self._file_name = name
        self._model = None
        self._aug = None

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_packages(['torch', "detectron2"])

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def pack(self, detectron_model):  # pylint:disable=arguments-differ
        try:
            import detectron2  # noqa # pylint: disable=unused-import
        except ImportError:
            raise MissingDependencyException(
                "Detectron package is required to use DetectronModelArtifact"
            )
        self._model = detectron_model
        return self

    def load(self, path):
        try:
            from detectron2.checkpoint import DetectionCheckpointer  # noqa # pylint: disable=unused-import
            from detectron2.modeling import META_ARCH_REGISTRY
            from detectron2.config import get_cfg
            from detectron2.data import transforms as T
            import json
        except ImportError:
            raise MissingDependencyException(
                "Detectron package is required to use DetectronArtifact"
            )

        cfg = get_cfg()
        cfg.merge_from_file(f"{path}/{self._file_name}.yaml")        
        meta_arch = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)
        self._model = meta_arch(cfg)
        self._model.eval()

        device = os.environ.get('BENTOML_DEVICE')
        if device == "GPU":
            device = "cuda:0"
        else:
            device = "cpu"
        self._model.to(device)
        checkpointer = DetectionCheckpointer(self._model)
        checkpointer.load(f"{path}/{self._file_name}.pth")
        self._aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        return self.pack(self._model)

    def get(self):
        return self._model

    def save(self, dst):
        try:
            from detectron2.checkpoint import DetectionCheckpointer  # noqa # pylint: disable=unused-import
            from detectron2.config import get_cfg
        except ImportError:
            raise MissingDependencyException(
                "Detectron package is required to use DetectronArtifact"
            )
        os.makedirs(dst, exist_ok = True)
        checkpointer = DetectionCheckpointer(self._model, save_dir=dst)
        checkpointer.save(self._file_name)
        cfg = get_cfg()
        cfg.merge_from_file("input_model.yaml")
        with open(os.path.join(dst, "model.yaml"), 'w', encoding='utf-8') as output_file:
            output_file.write(cfg.dump())
        


