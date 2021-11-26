from abc import ABC
from typing import Callable, Iterable, List
import torch
from .file_utils import add_start_docstrings

logger = ...
LOGITS_PROCESSOR_INPUTS_DOCSTRING = ...

class LogitsProcessor(ABC):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class LogitsWarper(ABC):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class LogitsProcessorList(list):
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor: ...

class MinLengthLogitsProcessor(LogitsProcessor):
    def __init__(self, min_length: int, eos_token_id: int) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class TemperatureLogitsWarper(LogitsWarper):
    def __init__(self, temperature: float) -> None: ...
    def __call__(
        self, input_ids: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor: ...

class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class TopPLogitsWarper(LogitsWarper):
    def __init__(
        self, top_p: float, filter_value: float = ..., min_tokens_to_keep: int = ...
    ) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class TopKLogitsWarper(LogitsWarper):
    def __init__(
        self, top_k: int, filter_value: float = ..., min_tokens_to_keep: int = ...
    ) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(self, ngram_size: int) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class EncoderNoRepeatNGramLogitsProcessor(LogitsProcessor):
    def __init__(
        self, encoder_ngram_size: int, encoder_input_ids: torch.LongTensor
    ) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class NoBadWordsLogitsProcessor(LogitsProcessor):
    def __init__(
        self, bad_words_ids: Iterable[Iterable[int]], eos_token_id: int
    ) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
    ) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class HammingDiversityLogitsProcessor(LogitsProcessor):
    def __init__(
        self, diversity_penalty: float, num_beams: int, num_beam_groups: int
    ) -> None: ...
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        current_tokens: torch.LongTensor,
        beam_group_idx: int,
    ) -> torch.FloatTensor: ...

class ForcedBOSTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, bos_token_id: int) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class ForcedEOSTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, max_length: int, eos_token_id: int) -> None: ...
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...

class InfNanRemoveLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor: ...
