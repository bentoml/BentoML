from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
from .file_utils import ModelOutput
from .generation_beam_search import BeamScorer
from .generation_logits_process import LogitsProcessorList
from .generation_stopping_criteria import StoppingCriteriaList

logger = ...

@dataclass
class GreedySearchDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = ...
    scores: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...

@dataclass
class GreedySearchEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = ...
    scores: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...

@dataclass
class SampleDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = ...
    scores: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...

@dataclass
class SampleEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = ...
    scores: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...

@dataclass
class BeamSearchDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = ...
    sequences_scores: Optional[torch.FloatTensor] = ...
    scores: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...

@dataclass
class BeamSearchEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = ...
    sequences_scores: Optional[torch.FloatTensor] = ...
    scores: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...

@dataclass
class BeamSampleDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = ...
    sequences_scores: Optional[torch.FloatTensor] = ...
    scores: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...

@dataclass
class BeamSampleEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = ...
    sequences_scores: Optional[torch.FloatTensor] = ...
    scores: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = ...

GreedySearchOutput = Union[
    GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput
]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]

class GenerationMixin:
    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, **kwargs
    ) -> Dict[str, Any]: ...
    def adjust_logits_during_generation(
        self, logits: torch.FloatTensor, **kwargs
    ) -> torch.FloatTensor: ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = ...,
        max_length: Optional[int] = ...,
        min_length: Optional[int] = ...,
        do_sample: Optional[bool] = ...,
        early_stopping: Optional[bool] = ...,
        num_beams: Optional[int] = ...,
        temperature: Optional[float] = ...,
        top_k: Optional[int] = ...,
        top_p: Optional[float] = ...,
        repetition_penalty: Optional[float] = ...,
        bad_words_ids: Optional[Iterable[int]] = ...,
        bos_token_id: Optional[int] = ...,
        pad_token_id: Optional[int] = ...,
        eos_token_id: Optional[int] = ...,
        length_penalty: Optional[float] = ...,
        no_repeat_ngram_size: Optional[int] = ...,
        encoder_no_repeat_ngram_size: Optional[int] = ...,
        num_return_sequences: Optional[int] = ...,
        max_time: Optional[float] = ...,
        max_new_tokens: Optional[int] = ...,
        decoder_start_token_id: Optional[int] = ...,
        use_cache: Optional[bool] = ...,
        num_beam_groups: Optional[int] = ...,
        diversity_penalty: Optional[float] = ...,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = ...,
        output_attentions: Optional[bool] = ...,
        output_hidden_states: Optional[bool] = ...,
        output_scores: Optional[bool] = ...,
        return_dict_in_generate: Optional[bool] = ...,
        forced_bos_token_id: Optional[int] = ...,
        forced_eos_token_id: Optional[int] = ...,
        remove_invalid_values: Optional[bool] = ...,
        synced_gpus: Optional[bool] = ...,
        **model_kwargs
    ) -> Union[
        GreedySearchOutput,
        SampleOutput,
        BeamSearchOutput,
        BeamSampleOutput,
        torch.LongTensor,
    ]: ...
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = ...,
        stopping_criteria: Optional[StoppingCriteriaList] = ...,
        max_length: Optional[int] = ...,
        pad_token_id: Optional[int] = ...,
        eos_token_id: Optional[int] = ...,
        output_attentions: Optional[bool] = ...,
        output_hidden_states: Optional[bool] = ...,
        output_scores: Optional[bool] = ...,
        return_dict_in_generate: Optional[bool] = ...,
        synced_gpus: Optional[bool] = ...,
        **model_kwargs
    ) -> Union[GreedySearchOutput, torch.LongTensor]: ...
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = ...,
        stopping_criteria: Optional[StoppingCriteriaList] = ...,
        logits_warper: Optional[LogitsProcessorList] = ...,
        max_length: Optional[int] = ...,
        pad_token_id: Optional[int] = ...,
        eos_token_id: Optional[int] = ...,
        output_attentions: Optional[bool] = ...,
        output_hidden_states: Optional[bool] = ...,
        output_scores: Optional[bool] = ...,
        return_dict_in_generate: Optional[bool] = ...,
        synced_gpus: Optional[bool] = ...,
        **model_kwargs
    ) -> Union[SampleOutput, torch.LongTensor]: ...
    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = ...,
        stopping_criteria: Optional[StoppingCriteriaList] = ...,
        max_length: Optional[int] = ...,
        pad_token_id: Optional[int] = ...,
        eos_token_id: Optional[int] = ...,
        output_attentions: Optional[bool] = ...,
        output_hidden_states: Optional[bool] = ...,
        output_scores: Optional[bool] = ...,
        return_dict_in_generate: Optional[bool] = ...,
        synced_gpus: Optional[bool] = ...,
        **model_kwargs
    ) -> Union[BeamSearchOutput, torch.LongTensor]: ...
    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = ...,
        stopping_criteria: Optional[StoppingCriteriaList] = ...,
        logits_warper: Optional[LogitsProcessorList] = ...,
        max_length: Optional[int] = ...,
        pad_token_id: Optional[int] = ...,
        eos_token_id: Optional[int] = ...,
        output_attentions: Optional[bool] = ...,
        output_hidden_states: Optional[bool] = ...,
        output_scores: Optional[bool] = ...,
        return_dict_in_generate: Optional[bool] = ...,
        synced_gpus: Optional[bool] = ...,
        **model_kwargs
    ) -> Union[BeamSampleOutput, torch.LongTensor]: ...
    def group_beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = ...,
        stopping_criteria: Optional[StoppingCriteriaList] = ...,
        max_length: Optional[int] = ...,
        pad_token_id: Optional[int] = ...,
        eos_token_id: Optional[int] = ...,
        output_attentions: Optional[bool] = ...,
        output_hidden_states: Optional[bool] = ...,
        output_scores: Optional[bool] = ...,
        return_dict_in_generate: Optional[bool] = ...,
        synced_gpus: Optional[bool] = ...,
        **model_kwargs
    ): ...

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = ...,
    top_p: float = ...,
    filter_value: float = ...,
    min_tokens_to_keep: int = ...,
) -> torch.FloatTensor: ...
