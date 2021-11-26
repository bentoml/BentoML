from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
from .file_utils import add_start_docstrings

PROCESS_INPUTS_DOCSTRING = ...
FINALIZE_INPUTS_DOCSTRING = ...

class BeamScorer(ABC):
    @abstractmethod
    @add_start_docstrings(PROCESS_INPUTS_DOCSTRING)
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        **kwargs
    ) -> Tuple[torch.Tensor]: ...
    @abstractmethod
    @add_start_docstrings(FINALIZE_INPUTS_DOCSTRING)
    def finalize(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        max_length: int,
        **kwargs
    ) -> torch.LongTensor: ...

class BeamSearchScorer(BeamScorer):
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = ...,
        do_early_stopping: Optional[bool] = ...,
        num_beam_hyps_to_keep: Optional[int] = ...,
        num_beam_groups: Optional[int] = ...,
        **kwargs
    ) -> None: ...
    @property
    def is_done(self) -> bool: ...
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id: Optional[int] = ...,
        eos_token_id: Optional[int] = ...,
    ) -> Tuple[torch.Tensor]: ...
    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int,
        pad_token_id: Optional[int] = ...,
        eos_token_id: Optional[int] = ...,
    ) -> Tuple[torch.LongTensor]: ...

class BeamHypotheses:
    def __init__(
        self, num_beams: int, length_penalty: float, early_stopping: bool
    ) -> None: ...
    def __len__(self): ...
    def add(self, hyp: torch.LongTensor, sum_logprobs: float): ...
    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool: ...
