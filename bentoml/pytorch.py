import typing as t
import zipfile
import functools
from typing import TYPE_CHECKING
from pathlib import Path

import cloudpickle
from simple_di import inject
from simple_di import Provide

from .exceptions import MissingDependencyException
from ._internal.types import Tag
from ._internal.models import Model
from ._internal.models import PT_EXT
from ._internal.models import SAVE_NAMESPACE
from ._internal.runner import Runner
from ._internal.runner.utils import Params
from ._internal.configuration.containers import BentoMLContainer

_RV = t.TypeVar("_RV")
_ModelType = t.TypeVar(
    "_ModelType", bound=t.Union["torch.nn.Module", "torch.jit.ScriptModule"]
)

if TYPE_CHECKING:
    from ._internal.models import ModelStore

try:
    import numpy as np
    import torch
    import torch.nn.parallel as parallel
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """
        torch is required in order to use module `bentoml.pytorch`.
         Instruction: Refers to https://pytorch.org/get-started/locally/
         to setup PyTorch correctly.
        """  # noqa
    )

infer_mode_compat = torch.__version__.startswith("1.9")


def _is_gpu_available() -> bool:  # pragma: no cover
    return torch.cuda.is_available()


@inject
def load(
    tag: t.Union[str, Tag],
    device_id: t.Optional[str] = "cpu",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _ModelType:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        device_id (`str`, `optional`):
            Optional devices to put the given model on. Refers to https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of either `torch.jit.ScriptModule` or `torch.nn.Module` from BentoML modelstore.

    Examples::
        import bentoml.pytorch
        booster = bentoml.pytorch.load(
            'lit_classifier:20201012_DE43A2', device_id="cuda:0")
    """  # noqa
    model = model_store.get(tag)
    weight_file = model.path_of(f"{SAVE_NAMESPACE}{PT_EXT}")
    # TorchScript Models are saved as zip files
    if zipfile.is_zipfile(weight_file):
        _load: t.Callable[[str], _ModelType] = functools.partial(
            torch.jit.load, map_location=device_id
        )
        return _load(weight_file)
    else:
        with Path(weight_file).open("rb") as file:
            __load: t.Callable[[t.BinaryIO], _ModelType] = functools.partial(
                cloudpickle.load
            )
            return __load(file)


@inject
def save(
    name: str,
    model: _ModelType,
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`t.Union[torch.nn.Module, torch.jit.ScriptModule]`):
            Instance of model to be saved
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
        import bentoml.pytorch
        import torch

        class NGramLanguageModeler(nn.Module):

            def __init__(self, vocab_size, embedding_dim, context_size):
                super(NGramLanguageModeler, self).__init__()
                self.embeddings = nn.Embedding(vocab_size, embedding_dim)
                self.linear1 = nn.Linear(context_size * embedding_dim, 128)
                self.linear2 = nn.Linear(128, vocab_size)

            def forward(self, inputs):
                embeds = self.embeddings(inputs).view((1, -1))
                out = F.relu(self.linear1(embeds))
                out = self.linear2(out)
                log_probs = F.log_softmax(out, dim=1)
                return log_probs

        tag = bentoml.pytorch.save("ngrams", NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE))
        # example tag: ngrams:20201012_DE43A2

    Integration with Torch Hub and BentoML::
        import bentoml.pytorch
        import torch

        resnet50 = torch.hub.load("pytorch/vision", "resnet50", pretrained=True)
        ...
        # trained a custom resnet50

        tag = bentoml.pytorch.save("resnet50", resnet50)
    """  # noqa
    context: t.Dict[str, t.Any] = dict(torch=torch.__version__)
    _model = Model.create(
        name,
        module=__name__,
        options=None,
        framework_context=context,
        metadata=metadata,
    )
    weight_file = _model.path_of(f"{SAVE_NAMESPACE}{PT_EXT}")
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, weight_file)
    else:
        with open(weight_file, "wb") as file:
            cloudpickle.dump(model, file)

    _model.save(model_store)
    return _model.tag


class _PyTorchRunner(Runner):
    @inject
    def __init__(
        self,
        tag: t.Union[str, Tag],
        predict_fn_name: str,
        device_id: str,
        partial_kwargs: t.Optional[t.Dict[str, t.Any]],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        in_store_tag = model_store.get(tag).tag

        super().__init__(str(in_store_tag), resource_quota, batch_options)
        self._predict_fn_name = predict_fn_name
        self._model_store = model_store
        if "cuda" in device_id:
            try:
                _, dev = device_id.split(":")
                self.resource_quota.gpus = [dev]
            except ValueError:
                self.resource_quota.gpus = [
                    str(i) for i in range(torch.cuda.device_count())
                ]
        self._device_id = device_id
        self._partial_kwargs = partial_kwargs or dict()
        self._tag = in_store_tag

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        if _is_gpu_available() and self.resource_quota.on_gpu:
            return 1
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        if _is_gpu_available() and self.resource_quota.on_gpu:
            return torch.cuda.device_count()
        return 1

    def _configure(self) -> None:
        torch.set_num_threads(self.num_concurrency_per_replica)
        if self.resource_quota.on_gpu and _is_gpu_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    @torch.no_grad()
    def _setup(self) -> None:  # type: ignore[override]
        self._configure()
        if self.resource_quota.on_gpu and _is_gpu_available():
            self._model = parallel.DataParallel(
                load(
                    self._tag,
                    model_store=self._model_store,
                    device_id=self._device_id,
                ),
            )
            torch.cuda.empty_cache()
        else:
            self._model = load(
                self._tag,
                model_store=self._model_store,
                device_id=self._device_id,
            )
        raw_predict_fn = getattr(self._model, self._predict_fn_name)
        self._predict_fn: t.Callable[..., _RV] = functools.partial(
            raw_predict_fn, **self._partial_kwargs
        )

    # pylint: disable=arguments-differ
    @torch.no_grad()
    def _run_batch(  # type: ignore[override]
        self,
        *args: t.Union[np.ndarray, torch.Tensor],
        **kwargs: str,
    ) -> torch.Tensor:

        params = Params[t.Any](*args, **kwargs)

        def _mapping(item) -> t.Any:
            if isinstance(item, np.ndarray):
                item = torch.from_numpy(item)

            if self.resource_quota.on_gpu:
                if isinstance(item, torch.Tensor):
                    item = item.cuda()

            return item

        params = params.map(_mapping)

        res: torch.Tensor
        if infer_mode_compat:
            with torch.inference_mode():
                res = self._predict_fn(*params.args, **kwargs)
        else:
            res = self._predict_fn(*params.args, **kwargs)
        return res


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    predict_fn_name: str = "__call__",
    device_id: str = "cpu:0",
    partial_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_PyTorchRunner":
    """
        Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.pytorch.load_runner` implements a Runner class that
    wrap around a pytorch instance, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        predict_fn_name (`str`, default to `__call__`):
            inference function to be used.
        device_id (`t.Union[str, int, t.List[t.Union[str, int]]]`, `optional`, default to `cpu`):
            Optional devices to put the given model on. Refers to https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
        partial_kwargs (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Common kwargs passed to model for this runner
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.pytorch` model

    Examples::
        import bentoml.pytorch
        runner = bentoml.pytorch.load_runner("ngrams:20201012_DE43A2")
        runner.run(pd.DataFrame("/path/to/csv"))
    """  # noqa
    return _PyTorchRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        device_id=device_id,
        partial_kwargs=partial_kwargs,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
