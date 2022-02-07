import pickle
import typing as t
import zipfile
import functools
from typing import TYPE_CHECKING
from pathlib import Path

import cloudpickle
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml import Runner

from ..types import LazyType
from ..models import PT_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..runner.utils import Params
from ..runner.container import Payload
from ..runner.container import DataContainer
from ..runner.container import DataContainerRegistry
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ..models import ModelStore

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

_ModelType = t.Union["torch.nn.Module", "torch.jit.ScriptModule"]  # type: ignore[reportPrivateUsage]

MODULE_NAME = "bentoml.pytorch"


def _is_gpu_available() -> bool:  # pragma: no cover
    return torch.cuda.is_available()


@inject
def load(
    tag: t.Union[Tag, str],
    device_id: t.Optional[str] = "cpu",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> _ModelType:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        device_id (:code:`str`, `optional`, default to :code:`cpu`):
            Optional devices to put the given model on. Refers to `device attributes <https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device>`_.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`Union[torch.jit.ScriptModule, torch.nn.Module]`: an instance of either :code:`torch.jit.ScriptModule` or :code:`torch.nn.Module` from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml
        model = bentoml.pytorch.load('lit_classifier:latest', device_id="cuda:0")
    """  # noqa
    bentoml_model = model_store.get(tag)
    if bentoml_model.info.module not in (MODULE_NAME, __name__):
        import bentoml._internal.frameworks.pytorch_lightning as pl

        if bentoml_model.info.module not in (pl.MODULE_NAME, pl.__name__):
            raise BentoMLException(
                f"Model {tag} was saved with module {bentoml_model.info.module}, failed loading with {MODULE_NAME}."
            )
    weight_file = bentoml_model.path_of(f"{SAVE_NAMESPACE}{PT_EXT}")
    model_format = bentoml_model.info.context.get("model_format")
    # backward compatibility
    if not model_format:
        if zipfile.is_zipfile(weight_file):
            model_format = "torchscript:v1"
        else:
            model_format = "cloudpickle:v1"

    if model_format == "torchscript:v1":
        model: "torch.jit.ScriptModule" = torch.jit.load(weight_file, map_location=device_id)  # type: ignore[reportPrivateImportUsage] # noqa: LN001
    elif model_format == "cloudpickle:v1":
        with Path(weight_file).open("rb") as file:
            model: "torch.nn.Module" = cloudpickle.load(file).to(device_id)
    elif model_format == "torch.save:v1":
        with Path(weight_file).open("rb") as file:
            model: "torch.nn.Module" = torch.load(file, map_location=device_id)
    else:
        raise BentoMLException(f"Unknown model format {model_format}")

    return model


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
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[torch.nn.Module, torch.jit.ScriptModule]`):
            Instance of model to be saved
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import torch
        import bentoml

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

    Integration with Torch Hub and BentoML:

    .. code-block:: python

        import torch
        import bentoml

        resnet50 = torch.hub.load("pytorch/vision", "resnet50", pretrained=True)
        ...
        # trained a custom resnet50

        tag = bentoml.pytorch.save("resnet50", resnet50)
    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "torch",
        "pip_dependencies": [f"torch=={get_pkg_version('torch')}"],
    }
    _model = Model.create(
        name,
        module=MODULE_NAME,
        options=None,
        context=context,
        metadata=metadata,
    )
    weight_file = _model.path_of(f"{SAVE_NAMESPACE}{PT_EXT}")
    if isinstance(model, torch.jit.ScriptModule):  # type: ignore[reportPrivateUsage]
        _model.info.context["model_format"] = "torchscript:v1"
        torch.jit.save(model, weight_file)  # type: ignore[reportUnknownMemberType]
    else:
        _model.info.context["model_format"] = "torch.save:v1"
        with open(weight_file, "wb") as file:
            torch.save(model, file, pickle_module=cloudpickle)

    _model.save(model_store)
    return _model.tag


class _PyTorchRunner(Runner):
    @inject
    def __init__(
        self,
        tag: Tag,
        predict_fn_name: str,
        device_id: str,
        name: str,
        partial_kwargs: t.Optional[t.Dict[str, t.Any]],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        in_store_tag = model_store.get(tag).tag

        super().__init__(name, resource_quota, batch_options)
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
    def _num_threads(self) -> int:
        if _is_gpu_available() and self.resource_quota.on_gpu:
            return 1
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        if _is_gpu_available() and self.resource_quota.on_gpu:
            return torch.cuda.device_count()
        return 1

    def _configure(self) -> None:
        torch.set_num_threads(self._num_threads)
        if self.resource_quota.on_gpu and _is_gpu_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    @torch.no_grad()
    def _setup(self) -> None:  # type: ignore[override]
        self._configure()
        model = load(
            self._tag, model_store=self._model_store, device_id=self._device_id
        )
        model.eval()
        if self.resource_quota.on_gpu and _is_gpu_available():
            self._model = parallel.DataParallel(model)
            torch.cuda.empty_cache()
        else:
            self._model = model
        raw_predict_fn = getattr(self._model, self._predict_fn_name)
        self._predict_fn: t.Callable[..., torch.Tensor] = functools.partial(
            raw_predict_fn, **self._partial_kwargs
        )

    @torch.no_grad()
    def _run_batch(
        self,
        *args: t.Union["np.ndarray[t.Any, np.dtype[t.Any]]", torch.Tensor],
        **kwargs: t.Union["np.ndarray[t.Any, np.dtype[t.Any]]", torch.Tensor],
    ) -> torch.Tensor:

        params = Params[t.Union["np.ndarray[t.Any, np.dtype[t.Any]]", torch.Tensor]](
            *args, **kwargs
        )

        def _mapping(
            item: t.Union["np.ndarray[t.Any, np.dtype[t.Any]]", torch.Tensor]
        ) -> torch.Tensor:
            if isinstance(item, np.ndarray):
                item = torch.from_numpy(item)
            if self.resource_quota.on_gpu:
                item = item.cuda()
            return item

        params = params.map(_mapping)

        # inference mode is required for PyTorch version 1.9.*
        if get_pkg_version("torch").startswith("1.9"):
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
    device_id: str = "cpu",
    partial_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    name: t.Optional[str] = None,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_PyTorchRunner":
    """
        Runner represents a unit of serving logic that can be scaled horizontally to
        maximize throughput. `bentoml.pytorch.load_runner` implements a Runner class that
        wrap around a pytorch instance, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        predict_fn_name (:code:`str`, default to :code:`__call__`):
            inference function to be used.
        device_id (:code:`str`, `optional`, default to :code:`cpu`): Optional devices to put the given model on. Refers to `device attributes <https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device>`_.
         to put the given model on. Refers to `device attributes <https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device>`_.
        partial_kwargs (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Common kwargs passed to model for this runner
        resource_quota (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        batch_options (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.pytorch` model

    Examples:

    .. code-block:: python

        import bentoml
        import pandas as pd

        runner = bentoml.pytorch.load_runner("ngrams:latest")
        runner.run(pd.DataFrame("/path/to/csv"))
    """  # noqa
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _PyTorchRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        device_id=device_id,
        partial_kwargs=partial_kwargs,
        name=name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )


class PytorchTensorContainer(DataContainer[torch.Tensor, torch.Tensor]):
    @classmethod
    def singles_to_batch(cls, singles, batch_axis=0):
        return torch.stack(singles, dim=batch_axis)

    @classmethod
    def batch_to_singles(cls, batch, batch_axis=0):
        return [
            torch.squeeze(tensor, dim=batch_axis)
            for tensor in torch.split(batch, 1, dim=batch_axis)
        ]

    @classmethod
    @inject
    def single_to_payload(
        cls,
        single,
    ) -> Payload:
        return cls.create_payload(
            pickle.dumps(single),
            {"plasma": False},
        )

    @classmethod
    @inject
    def payload_to_single(
        cls,
        payload: Payload,
    ):
        return pickle.loads(payload.data)

    batch_to_payload = single_to_payload
    payload_to_batch = payload_to_single


DataContainerRegistry.register_container(
    LazyType("torch", "Tensor"),
    LazyType("torch", "Tensor"),
    PytorchTensorContainer,
)
