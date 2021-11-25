

from abc import ABC

import jaxlib.xla_extension as jax_xla

from .file_utils import add_start_docstrings

logger = ...
LOGITS_PROCESSOR_INPUTS_DOCSTRING = ...
class FlaxLogitsProcessor(ABC):
    """Abstract base class for all logit processors that can be applied during generation."""
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray) -> jax_xla.DeviceArray:
        """Flax method for processing logits."""
        ...
    


class FlaxLogitsWarper(ABC):
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray) -> jax_xla.DeviceArray:
        """Flax method for warping logits."""
        ...
    


class FlaxLogitsProcessorList(list):
    """
    This class can be used to create a list of :class:`~transformers.FlaxLogitsProcessor` or
    :class:`~transformers.FlaxLogitsWarper` to subsequently process a :obj:`scores` input tensor. This class inherits
    from list and adds a specific `__call__` method to apply each :class:`~transformers.FlaxLogitsProcessor` or
    :class:`~transformers.FlaxLogitsWarper` to the inputs.
    """
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int, **kwargs) -> jax_xla.DeviceArray:
        ...
    


class FlaxTemperatureLogitsWarper(FlaxLogitsWarper):
    r"""
    :class:`transformers.LogitsWarper` for temperature (exponential scaling output probability distribution).

    Args:
        temperature (:obj:`float`):
            The value used to module the logits distribution.
    """
    def __init__(self, temperature: float) -> None:
        ...
    
    def __call__(self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int) -> jax_xla.DeviceArray:
        ...
    


class FlaxTopPLogitsWarper(FlaxLogitsWarper):
    """
    :class:`transformers.LogitsWarper` that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <=
    prob_cut_off.

    Args:
        top_p (:obj:`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or higher are
            kept for generation.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """
    def __init__(self, top_p: float, filter_value: float = ..., min_tokens_to_keep: int = ...) -> None:
        ...
    
    def __call__(self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int) -> jax_xla.DeviceArray:
        ...
    


class FlaxTopKLogitsWarper(FlaxLogitsWarper):
    r"""
    :class:`transformers.LogitsWarper` that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (:obj:`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (:obj:`float`, `optional`, defaults to :obj:`-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (:obj:`int`, `optional`, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """
    def __init__(self, top_k: int, filter_value: float = ..., min_tokens_to_keep: int = ...) -> None:
        ...
    
    def __call__(self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int) -> jax_xla.DeviceArray:
        ...
    


class FlaxForcedBOSTokenLogitsProcessor(FlaxLogitsProcessor):
    r"""
    :class:`~transformers.FlaxLogitsProcessor` that enforces the specified token as the first generated token.

    Args:
        bos_token_id (:obj:`int`):
            The id of the token to force as the first generated token.
    """
    def __init__(self, bos_token_id: int) -> None:
        ...
    
    def __call__(self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int) -> jax_xla.DeviceArray:
        ...
    


class FlaxForcedEOSTokenLogitsProcessor(FlaxLogitsProcessor):
    r"""
    :class:`~transformers.FlaxLogitsProcessor` that enforces the specified token as the last generated token when
    :obj:`max_length` is reached.

    Args:
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        eos_token_id (:obj:`int`):
            The id of the token to force as the last generated token when :obj:`max_length` is reached.
    """
    def __init__(self, max_length: int, eos_token_id: int) -> None:
        ...
    
    def __call__(self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int) -> jax_xla.DeviceArray:
        ...
    


class FlaxMinLengthLogitsProcessor(FlaxLogitsProcessor):
    r"""
    :class:`transformers.FlaxLogitsProcessor` enforcing a min-length by setting EOS probability to 0.

    Args:
        min_length (:obj:`int`):
            The minimum length below which the score of :obj:`eos_token_id` is set to :obj:`-float("Inf")`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """
    def __init__(self, min_length: int, eos_token_id: int) -> None:
        ...
    
    def __call__(self, input_ids: jax_xla.DeviceArray, scores: jax_xla.DeviceArray, cur_len: int) -> jax_xla.DeviceArray:
        ...
    


