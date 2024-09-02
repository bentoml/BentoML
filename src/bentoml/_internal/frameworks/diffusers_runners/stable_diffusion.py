from __future__ import annotations

import logging
import typing as t
from typing import TYPE_CHECKING

import diffusers
import torch

from .utils import get_model_or_download
from .utils import resolve_pipeline_class

if TYPE_CHECKING:
    import bentoml
    from bentoml._internal.frameworks.diffusers import LoraOptionType
    from bentoml._internal.frameworks.diffusers import TextualInversionOptionType


logger = logging.getLogger(__name__)

MODEL_NAME = "stable_diffusion"
MODEL_SHORT_NAME = "sd"
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-2-1"
PIPELINE_MAPPING = {
    "text2img": diffusers.StableDiffusionPipeline,
    "img2img": diffusers.StableDiffusionImg2ImgPipeline,
}
DEFAULT_SIZE = (768, 768)


def create_runner(
    model_id: str,
    *,
    name: str | None = None,
    use_available: bool = False,
    pipeline_class: (
        str | type[diffusers.DiffusionPipeline]
    ) = diffusers.StableDiffusionPipeline,
    scheduler_class: str | type[diffusers.SchedulerMixin] | None = None,
    torch_dtype: str | torch.dtype | None = None,
    enable_xformers: bool | None = None,
    enable_attention_slicing: int | str | None = None,
    enable_model_cpu_offload: bool | None = None,
    enable_sequential_cpu_offload: bool | None = None,
    enable_torch_compile: bool | None = None,
    low_cpu_mem_usage: bool | None = None,
    variant: str | None = None,
    load_pretrained_extra_kwargs: dict[str, t.Any] | None = None,
    lora_dir: str | None = None,
    lora_weights: LoraOptionType | list[LoraOptionType] | None = None,
    textual_inversions: (
        TextualInversionOptionType | list[TextualInversionOptionType] | None
    ) = None,
) -> bentoml.Runner:
    if isinstance(pipeline_class, str):
        pipeline_class = resolve_pipeline_class(pipeline_class, PIPELINE_MAPPING)

    model = get_model_or_download(
        MODEL_SHORT_NAME,
        model_id,
        use_available=use_available,
        pipeline_class=pipeline_class,
    )

    options = dict(
        pipeline_class=pipeline_class,
        scheduler_class=scheduler_class,
        torch_dtype=torch_dtype,
        enable_xformers=enable_xformers,
        enable_attention_slicing=enable_attention_slicing,
        enable_model_cpu_offload=enable_model_cpu_offload,
        enable_sequential_cpu_offload=enable_sequential_cpu_offload,
        enable_torch_compile=enable_torch_compile,
        low_cpu_mem_usage=low_cpu_mem_usage,
        variant=variant,
        load_pretrained_extra_kwargs=load_pretrained_extra_kwargs,
        lora_dir=lora_dir,
        lora_weights=lora_weights,
        textual_inversions=textual_inversions,
    )

    return model.with_options(**options).to_runner()
