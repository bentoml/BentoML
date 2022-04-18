import os
import re
import typing as t
import logging
from typing import TYPE_CHECKING
from distutils.dir_util import copy_tree

from simple_di import inject
from simple_di import Provide

import bentoml
from bentoml import Tag
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..utils import LazyLoader
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from .common.model_runner import BaseModelRunner
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

MODULE_NAME = "bentoml.paddle"

PADDLE_MODEL_EXTENSION = ".pdmodel"
PADDLE_PARAMS_EXTENSION = ".pdiparams"

_paddle_exc = """\
Instruction for installing `paddlepaddle`:
- CPU support only: `pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple`
- GPU support: (latest version): `python -m pip install paddlepaddle-gpu==2.1.3.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html`.
    For other version of CUDA or different platforms refer to https://www.paddlepaddle.org.cn/ for more information
"""

if TYPE_CHECKING:
    import numpy as np
    import paddle
    import paddle.nn
    import paddlehub as hub
    import paddle.inference
    import paddlehub.module.module as module
    from paddle.static import InputSpec
    from paddle.fluid.dygraph.dygraph_to_static.program_translator import StaticFunction

    from ..models import ModelStore

try:
    import paddle
    import paddle.nn
    import paddle.inference
    from paddle.fluid import core
except ImportError:  # pragma: no cover
    raise MissingDependencyException(_paddle_exc)

_hub_exc = (
    """\
`paddlehub` is required to use `bentoml.paddle.import_from_paddlehub()`. Make sure
 have `paddlepaddle` installed beforehand. Install `paddlehub` with
 `pip install paddlehub`.
"""
    + _paddle_exc
)

# TODO: supports for PIL.Image and pd.DataFrame?
# try:
#     import PIL
#     import PIL.Image
# except ImportError:
#     PIL = None
#     PIL.Image = None
# _PIL_warning = """\
# `Pillow` is optionally required to use `bentoml.paddle._PaddlePaddleRunner._run_batch`.
# Instruction: `pip install -U Pillow`
# """
#
# try:
#     import pandas as pd
# except ImportError:
#     pd = None
# _pd_warning = """\
# `pandas` is optionally required to use `bentoml.paddle._PaddlePaddleRunner._run_batch`.
# Instruction: `pip install -U pandas`
# """

hub = LazyLoader("hub", globals(), "paddlehub", exc_msg=_hub_exc)
manager = LazyLoader("manager", globals(), "paddlehub.module.manager", exc_msg=_hub_exc)
server = LazyLoader("server", globals(), "paddlehub.server.server", exc_msg=_hub_exc)
np = LazyLoader("np", globals(), "numpy")


def _clean_name(name: str) -> str:  # pragma: no cover
    return re.sub(r"\W|^(?=\d)-", "_", name)


@inject
def _load_paddle_bentoml_default_config(
    tag: t.Union[str, Tag],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "paddle.inference.Config":
    # https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/api/analysis_config.cc
    model = model_store.get(tag)
    config = paddle.inference.Config(
        model.path_of(f"{SAVE_NAMESPACE}{PADDLE_MODEL_EXTENSION}"),
        model.path_of(f"{SAVE_NAMESPACE}{PADDLE_PARAMS_EXTENSION}"),
    )
    config.enable_memory_optim()
    config.switch_ir_optim(True)
    return config


@inject
def load(
    tag: t.Union[str, Tag],
    config: t.Optional["paddle.inference.Config"] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **kwargs: str,
) -> t.Union[
    "paddle.inference.Predictor", t.Union["module.RunModule", "module.ModuleV1"]
]:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        config (`paddle.inference.Config`, `optional`, default to :code:`None`):
            Model config to be used to create a `Predictor` instance.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj::code:`Union[paddle.inference.Predictor, paddlehub.module.module.RunModule, paddlehub.module.module.ModuleV1]`: an instance of
        one of :obj:`paddle.inference.Predictor`, :obj:`paddlehub.module.module.RunModule`, and :obj:`paddlehub.module.module.ModuleV1`
        from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        model = bentoml.paddle.load(tag)

    """
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    if "paddlehub" in model.info.context:
        if model.info.options["from_local_dir"]:
            return hub.Module(directory=model.path)
        else:
            server.CacheUpdater(
                "update_cache",
                module=model.info.options["name"],
                version=model.info.options["version"],
            ).start()
            directory = (
                model.path
                if "_module_dir" not in model.info.options
                else model.path_of(model.info.options["_module_dir"])
            )
            return hub.Module(directory=directory, **kwargs)
    else:
        _config = _load_paddle_bentoml_default_config(tag) if not config else config
        return paddle.inference.create_predictor(_config)


def _save(
    name: str,
    model: t.Union[str, "paddle.nn.Layer", "paddle.inference.Predictor"],
    version: t.Optional[str],
    branch: t.Optional[str],
    source: t.Optional[str],
    update: bool,
    ignore_env_mismatch: bool,
    hub_module_home: t.Optional[str],
    keep_download_from_hub: bool,
    *,
    input_spec: t.Optional[t.Union[t.List["InputSpec"], t.Tuple["InputSpec", ...]]],
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]],
    model_store: "ModelStore",
) -> Tag:
    context: t.Dict[str, t.Any] = {
        "framework_name": "paddle",
        "pip_dependencies": [f"paddlepaddle=={get_pkg_version('paddlepaddle')}"],
    }
    if isinstance(model, str):
        context["paddlehub"] = hub.__version__
        if not os.path.isdir(model):
            try:  # pragma: no cover
                # NOTE: currently there is no need to test this feature.
                _model = model_store.get(model)
                if not keep_download_from_hub:
                    logger.warning(
                        f"""\
`{name}` is found under BentoML modelstore. Returning {_model.tag}...

Mechanism: `bentoml.paddle.import_from_paddlehub` will initialize an instance of
 `LocalModuleManager` from `paddlehub`. We will check whether the module is already
 downloaded under `paddlehub` cache, then copy over to BentoML modelstore.
The reason behind the design is due to internal mechanism of `hub.Module` with regarding
 inference. If we save the module in Paddle inference format, there is no guarantee that
 users can use Module functionality. Refers to https://paddlehub.readthedocs.io/en/release-v2.1/tutorial/custom_module.html
 for how a Module is structured.

For most use-case where you are using pretrained model provided by PaddlePaddle
 (https://www.paddlepaddle.org.cn/hublist), since {_model.tag} exists, this means you
 have previously imported the module from paddlehub to BentoML modelstore, we will
 return the existing tag.
For use-case where you have a custom `hub.Module` or wanting to use different iteration
 of the aforementioned pretrained model, specify `keep_download_from_hub=True`,
 `version=<your_specific_version>` or any other related `kwargs`. Refers to
 https://paddlehub.readthedocs.io/en/release-v2.1/api/module.html for more information.
                        """
                    )
                    _tag = _model.tag
                    server.CacheUpdater(
                        "update_cache", module=model, version=version
                    ).start()
                    return _tag
            except (FileNotFoundError, NotFound):
                pass

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        context=context,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
    ) as _model:
        if isinstance(model, str):
            # NOTE: for paddlehub there is no way to skip Module initialization,
            #  since `paddlehub` will always initialize a `hub.Module` regardless
            #  of any situation. Therefore, the bottleneck will only happen one time
            #  when users haven't saved the pretrained model under paddlehub cache
            #  directory.
            if not hub.server_check():
                raise BentoMLException("Unable to connect to PaddleHub server.")
            if os.path.isdir(model):
                directory = model
                target = _model.path

                _model.info.options["from_local_dir"] = True
            else:
                _local_manager = manager.LocalModuleManager(home=hub_module_home)
                user_module_cls = _local_manager.search(
                    name, source=source, branch=branch
                )
                if not user_module_cls or not user_module_cls.version.match(version):
                    user_module_cls = _local_manager.install(
                        name=name,
                        version=version,
                        source=source,
                        update=update,
                        branch=branch,
                        ignore_env_mismatch=ignore_env_mismatch,
                    )

                directory = _local_manager._get_normalized_path(user_module_cls.name)
                target = _model.path_of(user_module_cls.name)
                print(target)

                _model.info.options = {}
                _model.info.options.update(hub.Module.load_module_info(directory))
                _model.info.options["_module_dir"] = os.path.relpath(
                    target, _model.path
                )
                _model.info.options["from_local_dir"] = False
            copy_tree(directory, target)
        else:
            paddle.jit.save(
                model, _model.path_of(SAVE_NAMESPACE), input_spec=input_spec
            )

        return _model.tag


@inject
def save(
    name: str,
    model: t.Union["paddle.nn.Layer", "paddle.inference.Predictor", "StaticFunction"],
    *,
    input_spec: t.Optional[
        t.Union[t.List["InputSpec"], t.Tuple["InputSpec", ...]]
    ] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[paddle.nn.Layer, paddle.inference.Predictor, StaticFunction]`):
            Instance of :code:`paddle.nn.Layer`, decorated functions, or
            :code:`paddle.inference.Predictor` to be saved.
        input_spec (:code:`Union[List[InputSpec], Tuple[InputSpec, ...]]`, `optional`, default to :code:`None`):
            Describes the input of the saved model's forward method, which can be
            described by :obj:`InputSpec` or example Tensor. Moreover, we support to specify
            non-tensor type argument, such as `int`, `float`, `string`, or
            `list`/`dict` of them. If `None`, all input variables of the original
            Layer's forward method would be the inputs of the saved model. Generally
            this is **NOT RECOMMENDED** to use unless you know what you are doing.
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import random
        import numpy as np

        import bentoml
        import paddle
        import paddle.nn as nn
        from paddle.static import InputSpec

        IN_FEATURES = 13
        OUT_FEATURES = 1


        def set_random_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)


        class LinearModel(nn.Layer):
            def __init__(self):
                super(LinearModel, self).__init__()
                self.fc = nn.Linear(IN_FEATURES, OUT_FEATURES)

            @paddle.jit.to_static(input_spec=[InputSpec(shape=[IN_FEATURES], dtype="float32")])
            def forward(self, x):
                return self.fc(x)


        def train_paddle_model() -> "LinearModel":
            set_random_seed(SEED)
            model = LinearModel()
            loss = nn.MSELoss()
            adam = paddle.optimizer.Adam(parameters=model.parameters())

            train_data = paddle.text.datasets.UCIHousing(mode="train")

            loader = paddle.io.DataLoader(
                train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2
            )

            model.train()
            for _ in range(EPOCH_NUM):
                for _, (feature, label) in enumerate(loader()):
                    out = model(feature)
                    loss_fn = loss(out, label)
                    loss_fn.backward()
                    adam.step()
                    adam.clear_grad()
            return model

        model = train_paddle_model()
        # `save` a pretrained model to BentoML modelstore:
        tag = bentoml.paddle.save("linear_model", model, input_spec=InputSpec(shape=[IN_FEATURES], dtype="float32"))
    """
    return _save(
        name=name,
        model=model,
        input_spec=input_spec,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
        model_store=model_store,
        version=None,
        branch=None,
        source=None,
        update=False,
        hub_module_home=None,
        ignore_env_mismatch=False,
        keep_download_from_hub=False,
    )


@inject
def import_from_paddlehub(
    model_name: str,
    name: t.Optional[str] = None,
    version: t.Optional[str] = None,
    branch: t.Optional[str] = None,
    source: t.Optional[str] = None,
    update: bool = False,
    ignore_env_mismatch: bool = False,
    hub_module_home: t.Optional[str] = None,
    keep_download_from_hub: bool = False,
    *,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Import models from :code:`PaddleHub` and save it under BentoML modelstore.

    Args:
        model_name (:code:`str`):
            Name for a PaddleHub model. This can be either path to a Hub module or model
            name, for both v2 and v1.
        name (:code:`str`, `optional`, default to :code:`None`):
            Name for given model instance. This should pass Python identifier check. If
            not specified, then BentoML will save under :obj:`model_name`.
        version (:code:`str`, `optional`, default to :code:`None`):
            The version limit of the module, only takes effect when the :obj:`name` is
            specified. When the local Module does not meet the specified version
            conditions, PaddleHub will re-request the server to download the
            appropriate Module. Default to `None`, This means that the local Module will
            be used. If the Module does not exist, PaddleHub will download the latest
            version available from the server according to the usage environment.
        source (:code:`str`, `optional`, default to :code:`None`):
            URL of a git repository. If this parameter is specified, PaddleHub will no
            longer download the specified Module from the default server, but will look
            for it in the specified repository.
        update (`bool`, `optional`, default to :code:`False`):
            Whether to update the locally cached git repository, only takes effect when
            the `source` is specified.
        branch (:code:`str`, `optional`, default to :code:`None`):
            The branch of the specified git repository.
        ignore_env_mismatch (`bool`, `optional`, default to :code:`False`):
            Whether to ignore the environment mismatch when installing the Module.
        hub_module_home (:code:`str`, `optional`, default is `None`):
            Location to save for your PaddleHub cache. If `None`, then default to use
            PaddleHub default cache, which is under :code:`$HOME/.paddlehub`
        keep_download_from_hub (`bool`, `optional`, default to :code:`False`):
            Whether to re-download pretrained model from hub.
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import bentoml

        tag = bentoml.paddle.import_from_paddlehub("senta_bilstm")

    """
    _name = model_name.split("/")[-1] if os.path.isdir(model_name) else model_name

    return _save(
        name=_clean_name(_name) if not name else name,
        model=model_name,
        input_spec=None,
        version=version,
        branch=branch,
        source=source,
        update=update,
        ignore_env_mismatch=ignore_env_mismatch,
        hub_module_home=hub_module_home,
        keep_download_from_hub=keep_download_from_hub,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
        model_store=model_store,
    )


class _PaddlePaddleRunner(BaseModelRunner):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        infer_api_callback: str,
        *,
        gpu_mem_pool_mb: int = 1024,
        config: t.Optional["paddle.inference.Config"],
        name: t.Optional[str] = None,
    ):
        super().__init__(tag=tag, name=name)

        self._infer_api_callback = infer_api_callback
        self._config = config
        self._gpu_mem_pool_mb = gpu_mem_pool_mb

    @property
    def _enable_gpu(self):
        enable_gpu = self.resource_quota.on_gpu
        if enable_gpu and not paddle.is_compiled_with_cuda():  # type: ignore
            raise BentoMLException(
                "`resource_quota.on_gpu=True` while CUDA is not currently supported by existing paddlepaddle."
                " Make sure to install `paddlepaddle-gpu` and try again."
            )
        return enable_gpu

    def _setup_runner_config(self) -> None:
        _config = (
            _load_paddle_bentoml_default_config(self._tag)
            if not self._config
            else self._config
        )
        if self._enable_gpu:
            _config.enable_use_gpu(self._gpu_mem_pool_mb)
        else:
            # If not specific mkldnn, you can set the blas thread.
            # `num_threads` should not be greater than the number of cores in the CPU.
            _config.set_cpu_math_library_num_threads(self._num_threads)
            _config.enable_mkldnn()
            _config.disable_gpu()
        paddle.set_device(self._device)
        self._runner_config = _config

    @property
    def _device(self) -> str:
        if self._enable_gpu:
            # this will default to gpu:0
            return "gpu"
        return "cpu"

    @property
    def _num_threads(self) -> int:
        if self._enable_gpu and self.resource_quota.on_gpu:
            return 1
        return max(round(self.resource_quota.cpu), 1)

    @property
    def num_replica(self) -> int:
        if self._enable_gpu:
            count = len(self.resource_quota.gpus)
            return count
        return 1

    def _setup(self) -> None:
        self._setup_runner_config()
        self._model = load(self._tag, config=self._config, model_store=self.model_store)
        self._infer_func = getattr(self._model, self._infer_api_callback)

    def _run_batch(
        self,
        input_data: t.Optional[t.Any],
        *args: str,
        return_argmax: bool = False,
        **kwargs: str,
    ) -> t.Any:
        model_info = self.model_info
        if "paddlehub" in model_info.info.context:
            return self._infer_func(*args, **kwargs)
        else:
            assert input_data is not None
            res = list()
            if isinstance(input_data, paddle.Tensor):
                input_data = input_data.numpy()
            # TODO: supports for PIL.Image and DataFrame?
            # else:
            #     if PIL is not None:
            #         if isinstance(input_data, PIL.Image.Image):
            #             input_data = input_data.fromarray(input_data)
            #     else:
            #         logger.warning(_PIL_warning)
            #     if pd is not None:
            #         if isinstance(input_data, pd.DataFrame):
            #             input_data = input_data.to_numpy(dtype=np.float32)
            #         else:
            #             logger.warning(_pd_warning)

            input_names = self._model.get_input_names()

            for i, name in enumerate(input_names):
                input_tensor = self._model.get_input_handle(name)
                input_tensor.reshape(input_data[i].shape)
                input_tensor.copy_from_cpu(input_data[i].copy())

            self._infer_func()

            output_names = self._model.get_output_names()
            for i, name in enumerate(output_names):
                output_tensor = self._model.get_output_handle(name)
                output_data = output_tensor.copy_to_cpu()
                res.append(np.asarray(output_data))
            if return_argmax:
                return np.argmax(res, dims=-1)  # type: ignore[call-overload]
            return res


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    infer_api_callback: str = "run",
    gpu_mem_pool_mb: int = 0,
    name: t.Optional[str] = None,
    config: t.Optional["paddle.inference.Config"] = None,
) -> "_PaddlePaddleRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. :func:`bentoml.paddle.load_runner` implements a Runner class that
    either wraps around :obj:`paddle.inference.Predictor` or :obj:`paddlehub.module.Module`,
    which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        infer_api_callback (:code:`str`, `optional`, default to :code:`run`):
            Inference API function that will be used during :obj:`run_batch`. If :obj:`tag`
            is a tag of a :obj:`hub.Module`, then :code:`infer_api_callback` should be changed
            to corresponding API endpoint offered by given module.
        config (`paddle.inference.Config`, `optional`, default to :code:`None`):
            Config for inference. If None is specified, then use BentoML default.
        gpu_mem_pool_mb (`int`, `optional`, default to 0):
            Amount of memory one wants to allocate to GPUs. By default we will allocate None.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.paddle` model

    Examples:

    .. code-block:: python

        import bentoml
        import numpy as np

        runner = bentoml.paddle.load_runner(
            tag,
            device="gpu:0",
        )

        _ = runner.run_batch(pd_dataframe.to_numpy().astype(np.float32))

    """
    return _PaddlePaddleRunner(
        tag=tag,
        infer_api_callback=infer_api_callback,
        gpu_mem_pool_mb=gpu_mem_pool_mb,
        config=config,
        name=name,
    )
