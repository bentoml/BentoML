from typing import Dict, Optional
import flax
import jaxlib.xla_extension as jax_xla
from .file_utils import ModelOutput

logger = ...

@flax.struct.dataclass
class FlaxGreedySearchOutput(ModelOutput):
    sequences: jax_xla.DeviceArray = ...

@flax.struct.dataclass
class FlaxSampleOutput(ModelOutput):
    sequences: jax_xla.DeviceArray = ...

@flax.struct.dataclass
class FlaxBeamSearchOutput(ModelOutput):
    sequences: jax_xla.DeviceArray = ...
    scores: jax_xla.DeviceArray = ...

@flax.struct.dataclass
class GreedyState:
    cur_len: jax_xla.DeviceArray
    sequences: jax_xla.DeviceArray
    running_token: jax_xla.DeviceArray
    is_sent_finished: jax_xla.DeviceArray
    model_kwargs: Dict[str, jax_xla.DeviceArray]

...

@flax.struct.dataclass
class SampleState:
    cur_len: jax_xla.DeviceArray
    sequences: jax_xla.DeviceArray
    running_token: jax_xla.DeviceArray
    is_sent_finished: jax_xla.DeviceArray
    prng_key: jax_xla.DeviceArray
    model_kwargs: Dict[str, jax_xla.DeviceArray]

...

@flax.struct.dataclass
class BeamSearchState:
    cur_len: jax_xla.DeviceArray
    running_sequences: jax_xla.DeviceArray
    running_scores: jax_xla.DeviceArray
    sequences: jax_xla.DeviceArray
    scores: jax_xla.DeviceArray
    is_sent_finished: jax_xla.DeviceArray
    model_kwargs: Dict[str, jax_xla.DeviceArray]

...

class FlaxGenerationMixin:
    def generate(
        self,
        input_ids: jax_xla.DeviceArray,
        max_length: Optional[int] = ...,
        pad_token_id: Optional[int] = ...,
        bos_token_id: Optional[int] = ...,
        eos_token_id: Optional[int] = ...,
        decoder_start_token_id: Optional[int] = ...,
        do_sample: Optional[bool] = ...,
        prng_key: Optional[jax_xla.DeviceArray] = ...,
        top_k: Optional[int] = ...,
        top_p: Optional[float] = ...,
        temperature: Optional[float] = ...,
        num_beams: Optional[int] = ...,
        no_repeat_ngram_size: Optional[int] = ...,
        min_length: Optional[int] = ...,
        forced_bos_token_id: Optional[int] = ...,
        forced_eos_token_id: Optional[int] = ...,
        length_penalty: Optional[float] = ...,
        early_stopping: Optional[bool] = ...,
        trace: bool = ...,
        params: Optional[Dict[str, jax_xla.DeviceArray]] = ...,
        **model_kwargs
    ): ...
