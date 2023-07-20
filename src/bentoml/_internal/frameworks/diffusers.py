from __future__ import annotations

import logging
import os
import re
import shutil
import typing as t
from pathlib import Path
from typing import TYPE_CHECKING

import attr

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException
from bentoml.exceptions import NotFound
from bentoml.models import ModelContext

from ..models.model import PartialKwargsModelOptions

if TYPE_CHECKING:
    from types import ModuleType

    from bentoml.types import ModelSignature
    from bentoml.types import ModelSignatureDict


try:
    import diffusers
    import torch
    from diffusers.utils.import_utils import is_accelerate_available
    from diffusers.utils.import_utils import is_torch_version
    from diffusers.utils.import_utils import is_xformers_available
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'diffusers' and 'transformers' is required in order to use module 'bentoml.diffusers', install diffusers and its dependencies with 'pip install --upgrade diffusers transformers accelerate'. For more information, refer to https://github.com/huggingface/diffusers",
    )


MODULE_NAME = "bentoml.diffusers"
DIFFUSION_MODEL_FOLDER = "diffusion_model"
DIFFUSION_MODEL_CONFIG_FILE = "model_index.json"
API_VERSION = "v1"

logger = logging.getLogger(__name__)


@attr.define
class DiffusersOptions(PartialKwargsModelOptions):
    """Options for the diffusers model."""

    pipeline_class: str | type[diffusers.pipelines.DiffusionPipeline] | None = None
    scheduler_class: str | type[diffusers.SchedulerMixin] | None = None
    torch_dtype: str | torch.dtype | None = None
    device_map: str | dict[str, int | str | torch.device] | None = None
    custom_pipeline: str | None = None
    enable_xformers: bool | None = None
    enable_attention_slicing: int | str | None = None
    enable_model_cpu_offload: bool | None = None
    enable_sequential_cpu_offload: bool | None = None
    low_cpu_mem_usage: bool | None = None
    variant: str | None = None
    load_pretrained_extra_kwargs: dict[str, t.Any] | None = None


def _str2cls(
    full_cls_str: str,
) -> type[diffusers.pipelines.DiffusionPipeline | diffusers.SchedulerMixin]:
    import importlib

    module_name, _, class_name = full_cls_str.rpartition(".")

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def _extract_commit_hash(
    resolved_dir: str, regex_commit_hash: t.Pattern[str]
) -> str | None:
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    modified from https://github.com/huggingface/transformers/blob/0b7b4429c78de68acaf72224eb6dae43616d820c/src/transformers/utils/hub.py#L219
    """

    resolved_dir = str(Path(resolved_dir).as_posix()) + "/"
    search = re.search(r"snapshots/([^/]+)/", resolved_dir)

    if search is None:
        return None

    commit_hash = search.groups()[0]
    return commit_hash if regex_commit_hash.match(commit_hash) else None


def _try_import_huggingface_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:  # pragma: no cover
        raise MissingDependencyException(
            "'huggingface_hub' is required in order to download pretrained diffusion models, install with 'pip install huggingface-hub'. For more information, refer to https://huggingface.co/docs/huggingface_hub/quick-start",
        )


def get(tag_like: str | Tag) -> bentoml.Model:
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
       model = bentoml.diffusers.get("my_stable_diffusion_model")
    """
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(
    bento_model: str | Tag | bentoml.Model,
    device_id: str | torch.device | None = None,
    pipeline_class: type[
        diffusers.pipelines.DiffusionPipeline
    ] = diffusers.StableDiffusionPipeline,
    device_map: str | dict[str, int | str | torch.device] | None = None,
    custom_pipeline: str | None = None,
    scheduler_class: type[diffusers.SchedulerMixin] | None = None,
    torch_dtype: str | torch.dtype | None = None,
    low_cpu_mem_usage: bool | None = None,
    enable_xformers: bool = False,
    enable_attention_slicing: int | str | None = None,
    enable_model_cpu_offload: bool | None = None,
    enable_sequential_cpu_offload: bool | None = None,
    variant: str | None = None,
    load_pretrained_extra_kwargs: dict[str, t.Any] | None = None,
) -> diffusers.DiffusionPipeline:
    """
    Load a Diffusion model and convert it to diffusers `Pipeline <https://huggingface.co/docs/diffusers/api/pipelines/overview>`_
    with the given tag from the local BentoML model store.

    Args:
        bento_model:
            Either the tag of the model to get from the store, or a BentoML
            ``~bentoml.Model`` instance to load the model from.
        device_id (:code:`str`, `optional`, default to :code:`None`):
            Optional devices to put the given model on. Refer to `device attributes <https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device>`_.
        pipeline_class (:code:`type[diffusers.DiffusionPipeline]`, `optional`):
            DiffusionPipeline Class use to load the saved diffusion model, default to
            ``diffusers.StableDiffusionPipeline``. For more pipeline types, refer to
            `Pipeline Overview <https://huggingface.co/docs/diffusers/api/pipelines/overview>`_
        device_map (:code:`None | str | Dict[str, Union[int, str, torch.device]]`, `optional`):
            A map that specifies where each submodule should go. For more information, refer to
            `device_map <https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained.device_map>`_
        custom_pipeline (:code:`None | str`, `optional`):
            An identifier of custom pipeline hosted on github. For a list of community
            maintained custom piplines, refer to https://github.com/huggingface/diffusers/tree/main/examples/community
        scheduler_class (:code:`type[diffusers.SchedulerMixin]`, `optional`):
            Scheduler Class to be used by DiffusionPipeline
        torch_dtype (:code:`str | torch.dtype`, `optional`):
            Override the default `torch.dtype` and load the model under this dtype.
        low_cpu_mem_usage (:code:`bool`, `optional`):
            Speed up model loading by not initializing the weights and only loading the
            pre-trained weights. defaults to `True` if torch version >= 1.9.0 else `False`
        enable_xformers (:code:`bool`, `optional`):
            Use xformers optimization if it's available. For more info, refer to
            https://github.com/facebookresearch/xformers
        variant (`str`, *optional*):
            If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin.

    Returns:
        The Diffusion model loaded as diffusers pipeline from the BentoML model store.

    Example:

    .. code-block:: python

        import bentoml
        pipeline = bentoml.diffusers.load_model('my_diffusers_model:latest')
        pipeline(prompt)
    """  # noqa
    if not isinstance(bento_model, bentoml.Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )

    if pipeline_class is None:
        pipeline_class = diffusers.StableDiffusionPipeline

    diffusion_model_dir = bento_model.path_of(DIFFUSION_MODEL_FOLDER)

    if low_cpu_mem_usage is None:
        if is_torch_version(">=", "1.9.0") and is_accelerate_available():
            low_cpu_mem_usage = True
        else:
            low_cpu_mem_usage = False

    load_pretrained_extra_kwargs = load_pretrained_extra_kwargs or {}
    pipeline: diffusers.DiffusionPipeline = pipeline_class.from_pretrained(
        diffusion_model_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        device_map=device_map,
        custom_pipeline=custom_pipeline,
        variant=variant,
        **load_pretrained_extra_kwargs,
    )

    if scheduler_class:
        scheduler: diffusers.SchedulerMixin = scheduler_class.from_config(
            pipeline.scheduler.config
        )
        pipeline.scheduler = scheduler

    if device_id is not None:
        move_model_to_device = True

        if str(device_id).lower().startswith("cuda"):
            # when device_map is not None, we should not move the
            # pipeline to gpu again see
            # https://github.com/huggingface/diffusers/issues/2782
            if device_map is not None:
                move_model_to_device = False
            if enable_sequential_cpu_offload:
                move_model_to_device = False
            if enable_model_cpu_offload:
                move_model_to_device = False

        if move_model_to_device:
            pipeline = pipeline.to(device_id)

    if enable_xformers:
        pipeline.enable_xformers_memory_efficient_attention()

    if enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()

    if enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()

    if enable_attention_slicing is not None:
        pipeline.enable_attention_slicing(enable_attention_slicing)

    return pipeline


def import_model(
    name: Tag | str,
    model_name_or_path: str | os.PathLike[str],
    *,
    proxies: dict[str, str] | None = None,
    revision: str = "main",
    variant: str | None = None,
    pipeline_class: diffusers.pipelines.DiffusionPipeline | None = None,
    sync_with_hub_version: bool = False,
    signatures: dict[str, ModelSignatureDict | ModelSignature] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
    # ...
) -> bentoml.Model:
    """
    Import Diffusion model from a artifact URI to the BentoML model store.

    Args:
        name:
            The name to give to the model in the BentoML store. This must be a valid
            :obj:`~bentoml.Tag` name.
        model_name_or_path:
            Can be either:
            - A string, the *repo id* of a pretrained pipeline hosted inside a model repo on
              https://huggingface.co/ Valid repo ids have to be located under a user or organization name, like
              `CompVis/ldm-text2im-large-256`.
            - A path to a *directory* containing pipeline weights saved using
              [`~DiffusionPipeline.save_pretrained`], e.g., `./my_pipeline_directory/`.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        variant (`str`, *optional*):
            Variant of the model to import. For example there's "fp16" and "fp32" variant for "DeepFloyd/IF-I-XL-v1.0".
            This may save download bandwidth and local disk space.
        sync_with_hub_version (`bool`, default to False):
            If sync_with_hub_version is true, then the model imported by
        signatures:
            Signatures of predict methods to be used. If not provided, the signatures
            default to {"__call__": {"batchable": False}}. See
            :obj:`~bentoml.types.ModelSignature` for more details.
        labels:
            A default set of management labels to be associated with the model. For
            example: ``{"training-set": "data-v1"}``.
        custom_objects:
            Custom objects to be saved with the model. An example is
            ``{"my-normalizer": normalizer}``. Custom objects are serialized with
            cloudpickle.
        metadata:
            Metadata to be associated with the model. An example is ``{"param_a": .2}``.

            Metadata is intended for display in a model management UI and therefore all
            values in metadata dictionary must be a primitive Python type, such as
            ``str`` or ``int``.

    Returns:
        A :obj:`~bentoml.Model` instance referencing a saved model in the local BentoML
        model store.

    Example:

    .. code-block:: python

        import bentoml

        bentoml.diffusers.import_model(
            'my_sd15_model',
            "runwayml/stable-diffusion-v1-5",
            signatures={
                "__call__": {"batchable": False},
            }
        )
    """

    tag = Tag.from_taglike(name)

    if sync_with_hub_version:
        if tag.version is not None:
            logger.warn(
                f"sync_with_hub_version is True, user provided version {tag.version} may be overridden by huggingface hub's commit hash"
            )

    context = ModelContext(
        framework_name="diffusers",
        framework_versions={"diffusers": diffusers.__version__},
    )

    if signatures is None:
        signatures = {
            "__call__": {"batchable": False},
        }
        logger.info(
            'Using the default model signature for diffusers (%s) for model "%s".',
            signatures,
            name,
        )

    options_dict: dict[str, str] = {}
    if pipeline_class:
        cls_str = f"{pipeline_class.__module__}.{pipeline_class.__name__}"
        options_dict["pipeline_class"] = cls_str
    if variant:
        options_dict["variant"] = variant

    options = DiffusersOptions(**options_dict) if options_dict else None

    if os.path.isdir(model_name_or_path):
        src_dir = model_name_or_path
        if sync_with_hub_version:
            raise BentoMLException(
                "Cannot sync version with huggingface hub when importing a local model"
            )

    elif pipeline_class:
        _try_import_huggingface_hub()

        src_dir = pipeline_class.download(
            model_name_or_path, proxies=proxies, revision=revision, variant=variant
        )

        if sync_with_hub_version:
            from huggingface_hub.file_download import REGEX_COMMIT_HASH

            version = _extract_commit_hash(src_dir, REGEX_COMMIT_HASH)
            if version is not None:
                if variant is not None:
                    version = version + "-" + variant
                tag.version = version

    else:
        _try_import_huggingface_hub()
        from huggingface_hub import snapshot_download

        src_dir = snapshot_download(
            model_name_or_path,
            proxies=proxies,
            revision=revision,
        )

        if sync_with_hub_version:
            from huggingface_hub.file_download import REGEX_COMMIT_HASH

            version = _extract_commit_hash(src_dir, REGEX_COMMIT_HASH)
            if version is not None:
                tag.version = version

    with bentoml.models.create(
        tag,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        options=options,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
        context=context,
    ) as bento_model:
        diffusion_model_dir = bento_model.path_of(DIFFUSION_MODEL_FOLDER)
        ignore = shutil.ignore_patterns(".git")

        model_config_file = os.path.join(src_dir, DIFFUSION_MODEL_CONFIG_FILE)
        if not os.path.exists(model_config_file):
            raise BentoMLException(f'artifact "{src_dir}" is not a Diffusion model')

        shutil.copytree(src_dir, diffusion_model_dir, symlinks=False, ignore=ignore)

        return bento_model


def save_model(
    name: Tag | str,
    pipeline: diffusers.DiffusionPipeline,
    *,
    signatures: dict[str, ModelSignatureDict | ModelSignature] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save a DiffusionPipeline to the BentoML model store.

    Args:
        name:
            The name to give to the model in the BentoML store. This must be a valid
            :obj:`~bentoml.Tag` name.
        pipeline:
            Instance of the Diffusers pipeline to be saved
        signatures:
            Signatures of predict methods to be used. If not provided, the signatures
            default to {"__call__": {"batchable": False}}. See
            :obj:`~bentoml.types.ModelSignature` for more details.
        labels:
            A default set of management labels to be associated with the model. For
            example: ``{"training-set": "data-v1"}``.
        custom_objects:
            Custom objects to be saved with the model. An example is
            ``{"my-normalizer": normalizer}``. Custom objects are serialized with
            cloudpickle.
        metadata:
            Metadata to be associated with the model. An example is ``{"param_a": .2}``.

            Metadata is intended for display in a model management UI and therefore all
            values in metadata dictionary must be a primitive Python type, such as
            ``str`` or ``int``.

    Returns:
        A :obj:`~bentoml.Model` instance referencing a saved model in the local BentoML
        model store.

    """

    if not isinstance(pipeline, diffusers.DiffusionPipeline):
        raise BentoMLException(
            "'pipeline' must be an instance of 'diffusers.DiffusionPipeline'. "
        )

    context = ModelContext(
        framework_name="diffusers",
        framework_versions={"diffusers": diffusers.__version__},
    )

    if signatures is None:
        signatures = {
            "__call__": {"batchable": False},
        }
        logger.info(
            'Using the default model signature for diffusers (%s) for model "%s".',
            signatures,
            name,
        )

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        options=None,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
        context=context,
    ) as bento_model:
        diffusion_model_dir = bento_model.path_of(DIFFUSION_MODEL_FOLDER)
        pipeline.save_pretrained(diffusion_model_dir)

        return bento_model


def get_runnable(bento_model: bentoml.Model) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    bento_options = t.cast(DiffusersOptions, bento_model.info.options)
    partial_kwargs: dict[str, t.Any] = bento_options.partial_kwargs  # type: ignore
    pipeline_class: str | type[diffusers.DiffusionPipeline] = (
        bento_options.pipeline_class or diffusers.StableDiffusionPipeline
    )
    if isinstance(pipeline_class, str):
        pipeline_class = _str2cls(pipeline_class)

    scheduler_class: type[
        diffusers.SchedulerMixin
    ] | None = bento_options.scheduler_class

    custom_pipeline: str | None = bento_options.custom_pipeline
    _enable_xformers: bool | None = bento_options.enable_xformers
    enable_attention_slicing: int | str | None = bento_options.enable_attention_slicing
    enable_sequential_cpu_offload: bool | None = (
        bento_options.enable_sequential_cpu_offload
    )
    enable_model_cpu_offload: bool | None = bento_options.enable_model_cpu_offload
    low_cpu_mem_usage: bool | None = bento_options.low_cpu_mem_usage
    variant: str | None = bento_options.variant
    _torch_dtype: str | torch.dtype | None = bento_options.torch_dtype
    device_map: str | dict[
        str, int | str | torch.device
    ] | None = bento_options.device_map
    load_pretrained_extra_kwargs = bento_options.load_pretrained_extra_kwargs

    class DiffusersRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()

            if torch.cuda.is_available() and _torch_dtype is None:
                torch_dtype = torch.float16
            else:
                torch_dtype = _torch_dtype

            enable_xformers: bool = False
            if torch.cuda.is_available() and _enable_xformers is None:
                if is_xformers_available():
                    enable_xformers: bool = True

            device_id: str | None = None
            if torch.cuda.is_available():
                device_id = "cuda"

            self.pipeline: diffusers.DiffusionPipeline = load_model(
                bento_model,
                device_id=device_id,
                device_map=device_map,
                pipeline_class=pipeline_class,
                scheduler_class=scheduler_class,
                torch_dtype=torch_dtype,
                custom_pipeline=custom_pipeline,
                enable_xformers=enable_xformers,
                enable_attention_slicing=enable_attention_slicing,
                enable_sequential_cpu_offload=enable_sequential_cpu_offload,
                enable_model_cpu_offload=enable_model_cpu_offload,
                low_cpu_mem_usage=low_cpu_mem_usage,
                variant=variant,
                load_pretrained_extra_kwargs=load_pretrained_extra_kwargs,
            )

        @bentoml.Runnable.method(batchable=False)
        def _replace_scheduler(self, scheduler_txt: str):
            try:
                scheduler_cls = _str2cls(scheduler_txt)
                if isinstance(self.pipeline.scheduler, scheduler_cls):
                    return dict(success=True)
                if scheduler_cls in self.pipeline.scheduler.compatibles:
                    self.pipeline.scheduler = scheduler_cls.from_config(
                        self.pipeline.scheduler.config,
                    )
                    return dict(success=True)
                else:
                    return dict(
                        success=False,
                        error_message="scheduler class is incompatible to this pipeline",
                    )

            except (ModuleNotFoundError, ValueError, AttributeError):
                logger.info(f"Cannot import {scheduler_txt}")
                return dict(
                    success=False,
                    error_message="cannot import scheduler class",
                )

    def make_run_method(
        method_name: str, partial_kwargs: dict[str, t.Any] | None
    ) -> t.Callable[..., t.Any]:
        def _run_method(
            runnable_self: DiffusersRunnable,
            *args: t.Any,
            **kwargs: t.Any,
        ) -> t.Any:
            if method_partial_kwargs is not None:
                kwargs = dict(method_partial_kwargs, **kwargs)

            raw_method = getattr(runnable_self.pipeline, method_name)
            res = raw_method(*args, **kwargs)

            # handle BaseOutput cannot be serialized yet
            if isinstance(res, diffusers.utils.BaseOutput):
                res = res.to_tuple()

            return res

        return _run_method

    for method_name, options in bento_model.info.signatures.items():
        method_partial_kwargs = partial_kwargs.get(method_name)
        DiffusersRunnable.add_method(
            make_run_method(method_name, method_partial_kwargs),
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    return DiffusersRunnable
