from __future__ import annotations

from bentoml.exceptions import MissingDependencyException

try:
    import diffusers  # noqa
    import torch  # noqa
    from diffusers.loaders import LoraLoaderMixin as LoraLoaderMixin
    from diffusers.loaders import (
        TextualInversionLoaderMixin as TextualInversionLoaderMixin,
    )
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'diffusers' and 'transformers' are required in order to use module 'bentoml.diffusers_runners', install diffusers and its dependencies with 'pip install --upgrade diffusers transformers accelerate'. For more information, refer to https://github.com/huggingface/diffusers",
    )


from . import stable_diffusion as stable_diffusion
from . import stable_diffusion_xl as stable_diffusion_xl
