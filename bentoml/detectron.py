import os
import typing as t

import bentoml._internal.constants as const

from ._internal.models.base import MODEL_NAMESPACE, PTH_EXTENSION, YAML_EXTENSION, Model
from ._internal.types import MetadataType, PathType
from ._internal.utils import LazyLoader

_exc = const.IMPORT_ERROR_MSG.format(
    fwr="detectron2",
    module=__name__,
    inst="Refers to https://detectron2.readthedocs.io/en/latest/tutorials/install.html",  # noqa
)

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import detectron2.checkpoint as checkpoint
    import detectron2.config as config
    import detectron2.modeling as modeling
    import torch.nn as nn
else:
    checkpoint = LazyLoader(
        "checkpoint", globals(), "detectron2.checkpoint", exc_msg=_exc
    )
    config = LazyLoader("config", globals(), "detectron2.config", exc_msg=_exc)
    modeling = LazyLoader("modeling", globals(), "detectron2.modeling", exc_msg=_exc)


class DetectronModel(Model):
    """
    Model class for saving/loading :obj:`detectron2` models,
    in the form of :class:`~detectron2.checkpoint.DetectionCheckpointer`

    Args:
        model (`torch.nn.Module`):
            detectron2 model is of type :obj:`torch.nn.Module`
        input_model_yaml (`detectron2.config.CfgNode`, `optional`, default to `None`):
            model config from :meth:`detectron2.model_zoo.get_config_file`
        metadata (`Dict[str, Any]`, `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`detectron2` is required by DetectronModel
        InvalidArgument:
            model is not an instance of :class:`torch.nn.Module`

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:
    """

    def __init__(
        self,
        model: "nn.Module",
        input_model_yaml: t.Optional["config.CfgNode"] = None,
        metadata: t.Optional[MetadataType] = None,
    ):
        super(DetectronModel, self).__init__(model, metadata=metadata)
        self._input_model_yaml = input_model_yaml

    @classmethod
    def load(  # noqa # pylint: disable=arguments-differ
        cls, path: PathType, device: str = "cpu"
    ) -> "nn.Module":
        """
        Load a detectron model from given yaml path.

        Args:
            path (`Union[str, bytes, os.PathLike]`):
                Given path containing saved yaml
                 config for loading detectron model.
            device (`str`, `optional`, default to ``cpu``):
                Device type to cast model. Default behaviour similar
                 to :obj:`torch.device("cuda")` Options: "cuda" or "cpu".
                 If None is specified then return default config.MODEL.DEVICE

        Returns:
            :class:`torch.nn.Module`

        Raises:
            MissingDependencyException:
                ``detectron2`` is required by
                 :class:`~bentoml.detectron.DetectronModel`.
        """
        cfg: "config.CfgNode" = config.get_cfg()
        if device:
            cfg.MODEL.DEVICE = device

        weight_path = os.path.join(path, f"{MODEL_NAMESPACE}{PTH_EXTENSION}")
        yaml_path = os.path.join(path, f"{MODEL_NAMESPACE}{YAML_EXTENSION}")

        if os.path.isfile(yaml_path):
            cfg.merge_from_file(yaml_path)
        model: "nn.Module" = modeling.build_model(cfg)
        if device:
            model.to(device)

        model.eval()

        checkpointer: "checkpoint.DetectionCheckpointer" = (
            checkpoint.DetectionCheckpointer(model)
        )
        checkpointer.load(weight_path)
        return model

    def save(self, path: PathType) -> None:
        os.makedirs(path, exist_ok=True)
        checkpointer: "checkpoint.DetectionCheckpointer" = (
            checkpoint.DetectionCheckpointer(self._model, save_dir=path)
        )
        checkpointer.save(MODEL_NAMESPACE)
        if self._input_model_yaml:
            with open(
                os.path.join(path, f"{MODEL_NAMESPACE}{YAML_EXTENSION}"),
                "w",
                encoding="utf-8",
            ) as ouf:
                ouf.write(self._input_model_yaml.dump())
