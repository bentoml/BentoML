import os
import typing as t
from typing import TYPE_CHECKING

import numpy as np
from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import PTH_EXT, SAVE_NAMESPACE, YAML_EXT, Model
from ._internal.runner import Runner
from ._internal.runner.utils import Params
from ._internal.types import Tag
from .exceptions import BentoMLException, MissingDependencyException

if TYPE_CHECKING:  # pragma: no cover
    from ._internal.models import ModelStore
try:
    import detectron2.checkpoint as checkpoint
    import detectron2.config as config
    import detectron2.modeling as modeling
    import torch

except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """detectron2 is required in order to use module `bentoml.detectron`,
        install detectron2 with `pip install detectron2`. For more
        information, refers to
        https://detectron2.readthedocs.io/en/latest/tutorials/install.html
        """
    )


try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

_detectron2_version = importlib_metadata.version("detectron2")
_torch_version = importlib_metadata.version("torch")


@inject
def load(
    tag: t.Union[str, Tag],
    device: str = "cpu",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "nn.Module":
    """
    Load a model from BentoML local modelstore with given tag.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        device (`str`, `optional`, default to ``cpu``):
            Device type to cast model. Default behaviour similar
             to :obj:`torch.device("cuda")` Options: "cuda" or "cpu".
             If None is specified then return default config.MODEL.DEVICE
        model_store (`~bentoml._internal.models.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `torch.nn.Module`

    Examples::
        import bentoml.detectron
        model = bentoml.detectron.load(
            "my_detectron_model:20201012_DE43A2")
    """  # noqa

    model_info = model_store.get(tag)
    if model_info.info.module != __name__:
        raise BentoMLException(  # pragma: no cover
            f"Model {tag} was saved with"
            f" module {model_info.info.module},"
            f" failed loading with {__name__}."
        )

    cfg: config.CfgNode = config.get_cfg()

    weight_path = model_info.path_of(f"{SAVE_NAMESPACE}{PTH_EXT}")
    yaml_path = model_info.path_of(f"{SAVE_NAMESPACE}{YAML_EXT}")

    if os.path.isfile(yaml_path):
        cfg.merge_from_file(yaml_path)

    if device:
        cfg.MODEL.DEVICE = device

    model: "nn.Module" = modeling.build_model(cfg)
    if device:
        model.to(device)

    model.eval()

    checkpointer: checkpoint.DetectionCheckpointer = checkpoint.DetectionCheckpointer(
        model
    )

    checkpointer.load(weight_path)
    return model


@inject
def save(
    name: str,
    model: "torch.nn.Module",
    *,
    model_config: t.Optional[config.CfgNode] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`torch.nn.Module`):
            Instance of detectron2 model to be saved.
        model_config (`detectron2.config.CfgNode`, `optional`, default to `None`):
            model config from :meth:`detectron2.model_zoo.get_config_file`
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
        import bentoml.detectron

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

        # example tag: my_detectron_model:20211018_B9EABF

        # load the model back:
        loaded = bentoml.detectron.load("my_detectron_model:latest") # or
        loaded = bentoml.detectron.load(tag)

    """  # noqa

    context: t.Dict[str, t.Any] = {
        "detectron2": _detectron2_version,
        "torch": _torch_version,
    }
    options: t.Dict[str, t.Any] = dict()

    _model = Model.create(
        name,
        module=__name__,
        options=options,
        framework_context=context,
        metadata=metadata,
    )

    checkpointer = checkpoint.DetectionCheckpointer(model, save_dir=_model.path)
    checkpointer.save(SAVE_NAMESPACE)
    if model_config:
        with open(
            _model.path_of(f"{SAVE_NAMESPACE}{YAML_EXT}"),
            "w",
            encoding="utf-8",
        ) as ouf:
            ouf.write(model_config.dump())

    _model.save(model_store)

    return _model.tag


class _DetectronRunner(Runner):
    @inject
    def __init__(
        self,
        tag: t.Union[str, Tag],
        predict_fn_name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)
        self._tag = tag
        self._predict_fn_name = predict_fn_name
        self._model_store = model_store

    @property
    def required_models(self) -> t.List[str]:
        return [self._tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        return 1

    @property
    def num_replica(self) -> int:
        if self.resource_quota.on_gpu:
            return len(self.resource_quota.gpus)
        return 1

    # pylint: disable=attribute-defined-outside-init
    def _setup(self) -> None:
        if self.resource_quota.on_gpu:
            device = "cuda"
        else:
            device = "cpu"
        self._model = load(self._tag, device, self._model_store)
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    def _run_batch(
        self,
        *args: t.Union["np.ndarray[t.Any, np.dtype[t.Any]]", torch.Tensor],
        **kwargs: t.Any,
    ) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
        params = Params[t.Union["np.ndarray[t.Any, np.dtype[t.Any]]", torch.Tensor]](
            *args, **kwargs
        )

        def _mapping(
            item: t.Union["np.ndarray[t.Any, np.dtype[t.Any]]", torch.Tensor]
        ) -> torch.Tensor:
            if isinstance(item, np.ndarray):
                return torch.from_numpy(item)
            return item

        params = params.map(_mapping)
        images = np.split(*params.args, params.args.shape[0], 0)
        images = [image.squeeze(axis=0) for image in images]

        inputs = [dict(image=image) for image in images]

        res = self._predict_fn(inputs, **params.kwargs)
        return np.asarray(res, dtype=object)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    predict_fn_name: str = "__call__",
    *,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _DetectronRunner:
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.detectron.load_runner` implements a Runner class that
    wrap around a torch.nn.Module model, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        predict_fn_name (`str`, default to `__call__`):
            Options for inference functions. Default to `__call__`
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.detectron` model

    Examples:
        TODO
    """  # noqa
    return _DetectronRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
