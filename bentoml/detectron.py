import os

from bentoml._internal.artifacts import BaseArtifact
from bentoml._internal.exceptions import MissingDependencyException


class DetectronModel(BaseArtifact):
    """
    Artifact class for saving/loading Detectron2 models,
    in the form of detectron2.checkpoint.DetectionCheckpointer

    Args:
        name (string): name of the artifact

    Raises:
        MissingDependencyException: detectron2 package is required for
        DetectronModelArtifact
        InvalidArgument: invalid argument type, model being packed must be instance of
            torch.nn.Module

    Example usage:

    >>> # Train model with data
    >>>
    >>>
    >>> import bentoml
    >>> from bentoml.adapters import ImageInput
    >>> from bentoml.frameworks.detectron import DetectronModelArtifact
    >>> from detectron2.data import transforms as T
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts([DetectronModelArtifact('model')])
    >>> class CocoDetectronService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=ImageInput(), batch=False)
    >>>     def predict(self, img: np.ndarray) -> Dict:
    >>>         _aug = T.ResizeShortestEdge(
    >>>            [800, 800], 1333
    >>>        )
    >>>        boxes = None
    >>>        scores = None
    >>>        pred_classes = None
    >>>        pred_masks= None
    >>>        try:
    >>>            original_image = img[:, :, ::-1]
    >>>            height, width = original_image.shape[:2]
    >>>            image = _aug.get_transform(original_image).
    >>>                             apply_image(original_image)
    >>>            image = torch.as_tensor(image.astype("float32").
    >>>                             transpose(2, 0, 1))
    >>>            inputs = {"image": image, "height": height, "width": width}
    >>>            predictions = self.artifacts.model([inputs])[0]
    >>>
    >>>
    >>> cfg = get_cfg()
    >>> cfg.merge_from_file("input_model.yaml")
    >>> meta_arch = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)
    >>> model = meta_arch(cfg)
    >>> model.eval()
    >>> device = "cuda:{}".format(0)
    >>> model.to(device)
    >>> checkpointer = DetectionCheckpointer(model)
    >>> checkpointer.load("output/model.pth")
    >>> metadata = {
    >>>     'device' : device
    >>> }
    >>> bento_svc = CocoDetectronService()
    >>> bento_svc.pack('model', model, metadata, "input_model.yaml")
    >>> saved_path = bento_svc.save()
    """

    def __init__(self, name):
        super().__init__(name)
        self._model = None
        self._aug = None
        self._input_model_yaml = None

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name)

    @classmethod
    def load(cls, path):
        try:
            from detectron2.checkpoint import (  # noqa # pylint: disable=unused-import
                DetectionCheckpointer,
            )
            from detectron2.config import get_cfg
            from detectron2.data import transforms as T
            from detectron2.modeling import META_ARCH_REGISTRY
        except ImportError:
            raise MissingDependencyException(
                "Detectron package is required to use DetectronArtifact"
            )
        cfg = get_cfg()
        cfg.merge_from_file(f"{path}/{self.name}.yaml")
        meta_arch = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)
        self._model = meta_arch(cfg)
        self._model.eval()

        device = self._metadata["device"]
        self._model.to(device)
        checkpointer = DetectionCheckpointer(self._model)
        checkpointer.load(f"{path}/{self.name}.pth")
        self._aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        return self.pack(self._model)

    def save(self, dst):
        try:
            from detectron2.checkpoint import (  # noqa # pylint: disable=unused-import
                DetectionCheckpointer,
            )
            from detectron2.config import get_cfg
        except ImportError:
            raise MissingDependencyException(
                "Detectron package is required to use DetectronArtifact"
            )
        os.makedirs(dst, exist_ok=True)
        checkpointer = DetectionCheckpointer(self._model, save_dir=dst)
        checkpointer.save(self.name)
        cfg = get_cfg()
        cfg.merge_from_file(self._input_model_yaml)
        with open(
            os.path.join(dst, f"{self.name}.yaml"), "w", encoding="utf-8"
        ) as output_file:
            output_file.write(cfg.dump())
