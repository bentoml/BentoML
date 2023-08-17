from __future__ import annotations

import logging
import os
import typing as t
from typing import TYPE_CHECKING

try:
    import diffusers
    import torch
    from diffusers.loaders import LoraLoaderMixin as LoraLoaderMixin
    from diffusers.loaders import TextualInversionLoaderMixin as TextualInversionLoaderMixin
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'diffusers' and 'transformers' is required in order to use module 'bentoml.diffusers_runners', install diffusers and its dependencies with 'pip install --upgrade diffusers transformers accelerate'. For more information, refer to https://github.com/huggingface/diffusers",
    )


from . import stable_diffusion as stable_diffusion
from . import stable_diffusion_xl as stable_diffusion_xl
