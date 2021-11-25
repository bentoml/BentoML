

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from .file_utils import add_start_docstrings

PROCESS_INPUTS_DOCSTRING = ...
FINALIZE_INPUTS_DOCSTRING = ...
class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for :meth:`~transformers.PreTrainedModel.beam_search` and
    :meth:`~transformers.PreTrainedModel.beam_sample`.
    """
    @abstractmethod
    @add_start_docstrings(PROCESS_INPUTS_DOCSTRING)
    def process(self, input_ids: torch.LongTensor, next_scores: torch.FloatTensor, next_tokens: torch.LongTensor, next_indices: torch.LongTensor, **kwargs) -> Tuple[torch.Tensor]:
        ...
    
    @abstractmethod
    @add_start_docstrings(FINALIZE_INPUTS_DOCSTRING)
    def finalize(self, input_ids: torch.LongTensor, next_scores: torch.FloatTensor, next_tokens: torch.LongTensor, next_indices: torch.LongTensor, max_length: int, **kwargs) -> torch.LongTensor:
        ...
    


class BeamSearchScorer(BeamScorer):
    r"""
    :class:`transformers.BeamScorer` implementing standard beam search decoding.

    Adapted in part from `Facebook's XLM beam search code
    <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

    Reference for the diverse beam search algorithm and implementation `Ashwin Kalyan's DBS implementation
    <https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua>`__

    Args:
        batch_size (:obj:`int`):
            Batch Size of :obj:`input_ids` for which standard beam search decoding is run in parallel.
        max_length (:obj:`int`):
            The maximum length of the sequence to be generated.
        num_beams (:obj:`int`):
            Number of beams for beam search.
        device (:obj:`torch.device`):
            Defines the device type (*e.g.*, :obj:`"cpu"` or :obj:`"cuda"`) on which this instance of
            :obj:`BeamSearchScorer` will be allocated.
        length_penalty (:obj:`float`, `optional`, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
            model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
            sequences.
        do_early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beam_hyps_to_keep (:obj:`int`, `optional`, defaults to 1):
            The number of beam hypotheses that shall be returned upon calling
            :meth:`~transformer.BeamSearchScorer.finalize`.
        num_beam_groups (:obj:`int`):
            Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
            beams. See `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
    """
    def __init__(self, batch_size: int, num_beams: int, device: torch.device, length_penalty: Optional[float] = ..., do_early_stopping: Optional[bool] = ..., num_beam_hyps_to_keep: Optional[int] = ..., num_beam_groups: Optional[int] = ..., **kwargs) -> None:
        ...
    
    @property
    def is_done(self) -> bool:
        ...
    
    def process(self, input_ids: torch.LongTensor, next_scores: torch.FloatTensor, next_tokens: torch.LongTensor, next_indices: torch.LongTensor, pad_token_id: Optional[int] = ..., eos_token_id: Optional[int] = ...) -> Tuple[torch.Tensor]:
        ...
    
    def finalize(self, input_ids: torch.LongTensor, final_beam_scores: torch.FloatTensor, final_beam_tokens: torch.LongTensor, final_beam_indices: torch.LongTensor, max_length: int, pad_token_id: Optional[int] = ..., eos_token_id: Optional[int] = ...) -> Tuple[torch.LongTensor]:
        ...
    


class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool) -> None:
        """
        Initialize n-best list of hypotheses.
        """
        ...
    
    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        ...
    
    def add(self, hyp: torch.LongTensor, sum_logprobs: float):
        """
        Add a new hypothesis to the list.
        """
        ...
    
    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        ...
    


