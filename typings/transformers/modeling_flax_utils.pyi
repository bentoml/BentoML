import os
from typing import Any, Callable, Dict, Set, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey
from .configuration_utils import PretrainedConfig
from .file_utils import PushToHubMixin
from .generation_flax_utils import FlaxGenerationMixin

logger = ...

def quick_gelu(x): ...

ACT2FN = ...

class FlaxPreTrainedModel(PushToHubMixin, FlaxGenerationMixin):
    config_class = ...
    base_model_prefix = ...
    def __init__(
        self,
        config: PretrainedConfig,
        module: nn.Module,
        input_shape: Tuple = ...,
        seed: int = ...,
        dtype: jnp.dtype = ...,
    ) -> None: ...
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> Dict: ...
    @property
    def config(self) -> PretrainedConfig: ...
    @property
    def module(self) -> nn.Module: ...
    @property
    def params(self) -> Union[Dict, FrozenDict]: ...
    @property
    def required_params(self) -> Set: ...
    @params.setter
    def params(self, params: Union[Dict, FrozenDict]): ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        dtype: jnp.dtype = ...,
        *model_args,
        **kwargs
    ): ...
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike[str]],
        params: Any = ...,
        push_to_hub: bool = ...,
        **kwargs: Any
    ) -> None: ...

def overwrite_call_docstring(
    model_class: object, docstring: str
) -> Callable[..., Callable[..., Any]]: ...
def append_call_sample_docstring(
    model_class, tokenizer_class, checkpoint, output_type, config_class, mask=...
): ...
