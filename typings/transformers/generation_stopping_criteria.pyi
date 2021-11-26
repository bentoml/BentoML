from abc import ABC
from typing import Optional
import torch
from .file_utils import add_start_docstrings

STOPPING_CRITERIA_INPUTS_DOCSTRING = ...

class StoppingCriteria(ABC):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs
    ) -> bool: ...

class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int) -> None: ...
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool: ...

class MaxNewTokensCriteria(StoppingCriteria):
    def __init__(self, start_length: int, max_new_tokens: int) -> None: ...
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool: ...

class MaxTimeCriteria(StoppingCriteria):
    def __init__(
        self, max_time: float, initial_timestamp: Optional[float] = ...
    ) -> None: ...
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool: ...

class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool: ...
    @property
    def max_length(self) -> Optional[int]: ...

def validate_stopping_criteria(
    stopping_criteria: StoppingCriteriaList, max_length: int
) -> StoppingCriteriaList: ...
