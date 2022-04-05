import os
import typing as t
from typing import TYPE_CHECKING

import numpy as np
from simple_di import inject
from simple_di import Provide

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import PTH_EXT
from ..models import YAML_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ..runner.utils import Params
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from .. import external_typing as ext
    from ..models import ModelStore
try:
    # pylint: disable=unused-import
    import torch
    import detectron2  # noqa F401
    import detectron2.config as config
    import detectron2.modeling as modeling
    import detectron2.checkpoint as checkpoint
    from detectron2.checkpoint import DetectionCheckpointer
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """detectron2 is required in order to use module `bentoml.detectron`,
        install detectron2 with `pip install detectron2`. For more
        information, refers to
        https://detectron2.readthedocs.io/en/latest/tutorials/install.html
        """
    )

MODULE_NAME = "bentoml.detectron"


@inject
def load(
    tag: t.Union[str, Tag],
    device: str = "cpu",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "torch.nn.Module":
    """
    Load a model from BentoML local modelstore with given tag.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        device (:code:`str`, `optional`, default to :code:`cpu`):
            Device type to cast model. Default behaviour similar to :obj:`torch.device("cuda")` Options: "cuda" or "cpu". If None is specified then return default config.MODEL.DEVICE
        model_store (`~bentoml._internal.models.ModelStore`, default to :code:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`torch.nn.Module`: an instance of `torch.nn.Module`

    Examples:

    .. code-block:: python

        import bentoml
        model = bentoml.detectron.load("my_detectron_model")
    """  # noqa: LN001

    model_info = model_store.get(tag)
    if model_info.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model_info.info.module}, failed loading with {MODULE_NAME}."
        )

    cfg: config.CfgNode = config.get_cfg()

    weight_path = model_info.path_of(f"{SAVE_NAMESPACE}{PTH_EXT}")
    yaml_path = model_info.path_of(f"{SAVE_NAMESPACE}{YAML_EXT}")

    if os.path.isfile(yaml_path):
        cfg.merge_from_file(yaml_path)

    if device:
        cfg.MODEL.DEVICE = device

    model: "torch.nn.Module" = modeling.build_model(cfg)
    if device:
        model.to(device)

    model.eval()

    checkpointer: "DetectionCheckpointer" = checkpoint.DetectionCheckpointer(model)

    checkpointer.load(weight_path)
    return model


def save(
    name: str,
    model: "torch.nn.Module",
    *,
    model_config: t.Optional[config.CfgNode] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`torch.nn.Module`):
            Instance of detectron2 model to be saved.
        model_config (`detectron2.config.CfgNode`, `optional`, default to :code:`None`):
            model config from :meth:`detectron2.model_zoo.get_config_file`
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.ModelStore`, default to :code:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import bentoml

        # import some common detectron2 utilities
        import detectron2
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model

        model_url: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        cfg: "CfgNode" = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_url))
        # set threshold for this model
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)
        cloned = cfg.clone()
        cloned.MODEL.DEVICE = "cpu"
        model: torch.nn.Module = build_model(cloned)

        tag = bentoml.detectron.save(
            "my_detectron_model",
            model,
            model_config=cfg,
        )

        # load the model back:
        loaded = bentoml.detectron.load("my_detectron_model:latest")
        # or:
        loaded = bentoml.detectron.load(tag)

    """  # noqa

    context: t.Dict[str, t.Any] = {
        "framework_name": "detectron2",
        "pip_dependencies": [
            f"detectron2=={get_pkg_version('detectron2')}",
            f"torch=={get_pkg_version('torch')}",
        ],
    }
    options: t.Dict[str, t.Any] = dict()

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        labels=labels,
        custom_objects=custom_objects,
        options=options,
        context=context,
        metadata=metadata,
    ) as _model:
        checkpointer = checkpoint.DetectionCheckpointer(model, save_dir=_model.path)
        checkpointer.save(SAVE_NAMESPACE)
        if model_config:
            with open(
                _model.path_of(f"{SAVE_NAMESPACE}{YAML_EXT}"),
                "w",
                encoding="utf-8",
            ) as ouf:
                ouf.write(model_config.dump())

        return _model.tag


from .common.model_runner import BaseModelRunner


class _DetectronRunner(BaseModelRunner):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        predict_fn_name: str,
        name: t.Optional[str] = None,
    ):
        super().__init__(tag, name=name)
        self._predict_fn_name = predict_fn_name

    @property
    def num_replica(self) -> int:
        if self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    @property
    def _device(self) -> str:
        if self.resource_quota.on_gpu:
            return "cuda"
        return "cpu"

    def _setup(self) -> None:
        self._model = load(self._tag, self._device, model_store=self.model_store)
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    def _run_batch(  # type: ignore
        self,
        *args: t.Union["ext.NpNDArray", torch.Tensor],
    ) -> "ext.NpNDArray":
        params = Params[t.Union["ext.NpNDArray", torch.Tensor]](*args)

        def _mapping(item: t.Union["ext.NpNDArray", torch.Tensor]) -> torch.Tensor:
            if isinstance(item, np.ndarray):
                return torch.Tensor(item, device=self._device)
            return item

        params = params.map(_mapping)

        inputs = [{"image": image} for image in params.args]

        res: "torch.Tensor" = self._predict_fn(inputs)
        return np.asarray(res)  # type: ignore


def load_runner(
    tag: t.Union[str, Tag],
    predict_fn_name: str = "__call__",
    *,
    name: t.Optional[str] = None,
) -> _DetectronRunner:
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. :func:`bentoml.detectron.load_runner` implements a Runner class that
    wrap around a :obj:`torch.nn.Module` model, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (:code:`str`, default to :code:`__call__`):
            Options for inference functions. Default to `__call__`
        model_store (`~bentoml._internal.models.ModelStore`, default to :code:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.detectron` model

    Examples:

    .. code-block:: python

        import bentoml
        import numpy as np

        runner = bentoml.detectron.load_runner(tag)
        runner.run_batch(np.array([[1,2,3,]]))
    """
    return _DetectronRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        name=name,
    )
