from abc import ABC
import jaxlib.xla_extension as jax_xla
from .file_utils import add_start_docstrings

logger = ...
LOGITS_PROCESSOR_INPUTS_DOCSTRING = ...

class FlaxLogitsProcessor(ABC):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray
    ) -> jax_xla.DeviceArray: ...

class FlaxLogitsWarper(ABC):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray
    ) -> jax_xla.DeviceArray: ...

class FlaxLogitsProcessorList(list):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: jax_xla.DeviceArray,
        scores: jax_xla.DeviceArray,
        cur_len: int,
        **kwargs
    ) -> jax_xla.DeviceArray: ...

class FlaxTemperatureLogitsWarper(FlaxLogitsWarper):
    def __init__(self, temperature: float) -> None: ...
    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int
    ) -> jax_xla.DeviceArray: ...

class FlaxTopPLogitsWarper(FlaxLogitsWarper):
    def __init__(
        self, top_p: float, filter_value: float = ..., min_tokens_to_keep: int = ...
    ) -> None: ...
    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int
    ) -> jax_xla.DeviceArray: ...

class FlaxTopKLogitsWarper(FlaxLogitsWarper):
    def __init__(
        self, top_k: int, filter_value: float = ..., min_tokens_to_keep: int = ...
    ) -> None: ...
    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int
    ) -> jax_xla.DeviceArray: ...

class FlaxForcedBOSTokenLogitsProcessor(FlaxLogitsProcessor):
    def __init__(self, bos_token_id: int) -> None: ...
    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int
    ) -> jax_xla.DeviceArray: ...

class FlaxForcedEOSTokenLogitsProcessor(FlaxLogitsProcessor):
    def __init__(self, max_length: int, eos_token_id: int) -> None: ...
    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int
    ) -> jax_xla.DeviceArray: ...

class FlaxMinLengthLogitsProcessor(FlaxLogitsProcessor):
    def __init__(self, min_length: int, eos_token_id: int) -> None: ...
    def __call__(
        self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int
    ) -> jax_xla.DeviceArray: ...
