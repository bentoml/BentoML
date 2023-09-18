from __future__ import annotations

import typing as t
import logging
from types import ModuleType

import bentoml

from ..tag import Tag
from ..types import LazyType
from ..utils import LazyLoader
from ..utils.pkg import get_pkg_version
from ...exceptions import NotFound
from ...exceptions import MissingDependencyException
from ..models.model import Model
from ..models.model import ModelContext
from ..models.model import ModelSignature
from ..models.model import PartialKwargsModelOptions as ModelOptions
from ..runner.utils import Params
from .common.pytorch import torch
from .common.pytorch import inference_mode_ctx
from .common.pytorch import PyTorchTensorContainer  # noqa # type: ignore

try:
    import detectron2.config as Config
    import detectron2.engine as Engine
    import detectron2.modeling as Modeling
    import detectron2.checkpoint as Checkpoint
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'detectron2' is required in order to use module 'bentoml.detectron'. Install detectron2 with 'pip install 'git+https://github.com/facebookresearch/detectron2.git''."
    )

if t.TYPE_CHECKING:
    import torch.nn as nn

    from .. import external_typing as ext
    from ..models.model import ModelSignaturesType
else:
    nn = LazyLoader("nn", globals(), "torch.nn")


__all__ = ["load_model", "save_model", "get_runnable", "get"]

MODULE_NAME = "bentoml.detectron"
API_VERSION = "v1"
MODEL_FILENAME = "saved_model"
DETECTOR_EXTENSION = ".pth"


logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get(tag_like: str | Tag) -> Model:
    """
    Get the BentoML model with the given tag.

    Args:
        tag_like: The tag of the model to retrieve from the model store.

    Returns:
        :obj:`~bentoml.Model`: A BentoML :obj:`~bentoml.Model` with the matching tag.

    Example:

    .. code-block:: python

       import bentoml
       # target model must be from the BentoML model store
       model = bentoml.detectron2.get("en_reader:latest")
    """
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(
    bento_model: str | Tag | Model, device_id: str = "cpu"
) -> Engine.DefaultPredictor | nn.Module:
    """
    Load the detectron2 model from BentoML local model store with given name.

    Args:
        bento_model: Either the tag of the model to get from the store,
                     or a BentoML :class:`~bentoml.Model` instance to load the
                     model from.
        device_id: The device to load the model to. Default to "cpu".

    Returns:
        One of the following:
        - ``detectron2.engine.DefaultPredictor`` if the the checkpointables is saved as a Predictor.
        - ``torch.nn.Module`` if the checkpointables is saved as a nn.Module

    Example:

    .. code-block:: python

        import bentoml
        predictor = bentoml.detectron2.load_model('predictor:latest')

        model = bentoml.detectron2.load_model('model:latest')
    """
    if not isinstance(bento_model, Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )

    cfg = bento_model.custom_objects["config"]
    cfg.MODEL.DEVICE = device_id

    metadata = bento_model.info.metadata
    if metadata.get("_is_predictor", True):
        return Engine.DefaultPredictor(cfg)
    else:
        model = Modeling.build_model(cfg)
        model.to(device).eval()
        Checkpoint.DetectionCheckpointer(model).load(
            bento_model.path_of(f"{MODEL_FILENAME}{DETECTOR_EXTENSION}")
        )
        return model


def save_model(
    name: Tag | str,
    checkpointables: Engine.DefaultPredictor | nn.Module,
    config: Config.CfgNode | None = None,
    *,
    signatures: ModelSignaturesType | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name: Name for given model instance. This should pass Python identifier check.
        checkpointables: The model instance to be saved. Could be a ``detectron2.engine.DefaultPredictor`` or a ``torch.nn.Module``.
        config: Optional ``CfgNode`` for the model. Required when checkpointables is a ``torch.nn.Module``.
        signatures: Methods to expose for running inference on the target model. Signatures are used for creating :obj:`~bentoml.Runner` instances when serving model with :obj:`~bentoml.Service`
        labels: User-defined labels for managing models, e.g. ``team=nlp``, ``stage=dev``.
        custom_objects: Custom objects to be saved with the model. An example is ``{"my-normalizer": normalizer}``.
                        Custom objects are currently serialized with cloudpickle, but this implementation is subject to change.
        external_modules: user-defined additional python modules to be saved alongside the model or custom objects,
                          e.g. a tokenizer module, preprocessor module, model configuration module
        metadata: Custom metadata for given model.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format ``name:version`` where ``name`` is the user-defined model's name, and a generated ``version``.

    Examples:

    .. tab-set::

       .. tab-item:: ModelZoo and CfgNode

          .. code-block:: python

              import bentoml
              import detectron2
              from detectron2 import model_zoo
              from detectron2.config import get_cfg
              from detectron2.modeling import build_model

              model_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
              cfg = get_cfg()
              cfg.merge_from_file(model_zoo.get_config_file(model_url))
              # set threshold for this model
              cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
              cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)
              cloned = cfg.clone()
              cloned.MODEL.DEVICE = "cpu"

              bento_model = bentoml.detectron2.save_model('mask_rcnn', build_model(cloned), config=cloned)

       .. tab-item:: Predictor

          .. code-block:: python

              import bentoml
              import detectron2
              from detectron2.engine import DefaultPredictor
              from detectron2 import model_zoo
              from detectron2.config import get_cfg
              from detectron2.modeling import build_model

              model_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
              cfg = get_cfg()
              cfg.merge_from_file(model_zoo.get_config_file(model_url))
              # set threshold for this model
              cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
              cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_url)
              cloned = cfg.clone()
              cloned.MODEL.DEVICE = "cpu"

              predictor = DefaultPredictor(cloned)
              bento_model = bentoml.detectron2.save_model('mask_rcnn', predictor)
    """  # noqa
    context = ModelContext(
        framework_name="detectron2",
        framework_versions={"detectron2": get_pkg_version("detectron2")},
    )

    if signatures is None:
        signatures = {"__call__": {"batchable": False}}
        logger.info(
            'Using the default model signature for Detectron (%s) for model "%s".',
            signatures,
            name,
        )
    if metadata is None:
        metadata = {}

    metadata["_is_predictor"] = isinstance(checkpointables, Engine.DefaultPredictor)

    if custom_objects is None:
        custom_objects = {}

    if isinstance(checkpointables, nn.Module):
        if config is None:
            raise ValueError(
                "config is required when 'checkpointables' is a derived 'torch.nn.Module'."
            )
        model = checkpointables
        model.eval()
    else:
        model = checkpointables.model
        if config is not None:
            logger.warning(
                "config is ignored when 'checkpointables' is a 'DefaultPredictor'."
            )
        config = t.cast(Config.CfgNode, checkpointables.cfg)

    custom_objects["config"] = config

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        labels=labels,
        context=context,
        options=ModelOptions(),
        signatures=signatures,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
    ) as bento_model:
        checkpointer = Checkpoint.Checkpointer(model, save_dir=bento_model.path)
        checkpointer.save(MODEL_FILENAME)
        return bento_model


def get_runnable(bento_model: bentoml.Model) -> type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    is_predictor = bento_model.info.metadata.get("_is_predictor", True)
    partial_kwargs = t.cast(ModelOptions, bento_model.info.options).partial_kwargs

    class Detectron2Runnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()
            if torch.cuda.is_available():
                self.device_id = "cuda"
                torch.set_default_tensor_type("torch.cuda.FloatTensor")
            else:
                self.device_id = "cpu"

            self.model = load_model(bento_model, device_id=self.device_id)

            if not is_predictor:
                assert isinstance(self.model, torch.nn.Module)
                # This predictor is a torch.nn.Module
                self.model.train(False)

            self.is_predictor = is_predictor

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in bento_model.info.signatures:
                self.predict_fns[method_name] = getattr(self.model, method_name)

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(self: Detectron2Runnable, *args: t.Any, **kwargs: t.Any) -> t.Any:
            method_partial_kwargs = partial_kwargs.get(method_name, {})

            params = Params(*args, **kwargs)

            if self.is_predictor:
                return self.predict_fns[method_name](
                    *params.args, **dict(method_partial_kwargs, **params.kwargs)
                )

            def mapping(item: ext.NpNDArray | torch.Tensor) -> t.Any:
                if LazyType["ext.NpNDArray"]("numpy.ndarray").isinstance(item):
                    return torch.Tensor(item, device=self.device_id)
                elif isinstance(item, torch.Tensor):
                    return item.to(self.device_id)
                else:
                    return item

            with inference_mode_ctx():
                params = params.map(mapping)
                return self.predict_fns[method_name](
                    [{"image": image} for image in params.args],
                    **dict(method_partial_kwargs, **params.kwargs),
                )

        Detectron2Runnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return Detectron2Runnable
