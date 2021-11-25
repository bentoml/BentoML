

from typing import Dict, Optional

import flax
import jaxlib.xla_extension as jax_xla

from .file_utils import ModelOutput

logger = ...
@flax.struct.dataclass
class FlaxGreedySearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, max_length)`):
            The generated sequences.
    """
    sequences: jax_xla.DeviceArray = ...


@flax.struct.dataclass
class FlaxSampleOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, max_length)`):
            The generated sequences.
    """
    sequences: jax_xla.DeviceArray = ...


@flax.struct.dataclass
class FlaxBeamSearchOutput(ModelOutput):
    """
    Flax Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, max_length)`):
            The generated sequences.
        scores (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size,)`):
            The scores (log probabilites) of the generated sequences.
    """
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
    """
    A class containing all of the functions supporting generation, to be used as a mixin in
    :class:`~transformers.FlaxPreTrainedModel`.
    """
    def generate(self, input_ids: jax_xla.DeviceArray, max_length: Optional[int] = ..., pad_token_id: Optional[int] = ..., bos_token_id: Optional[int] = ..., eos_token_id: Optional[int] = ..., decoder_start_token_id: Optional[int] = ..., do_sample: Optional[bool] = ..., prng_key: Optional[jax_xla.DeviceArray] = ..., top_k: Optional[int] = ..., top_p: Optional[float] = ..., temperature: Optional[float] = ..., num_beams: Optional[int] = ..., no_repeat_ngram_size: Optional[int] = ..., min_length: Optional[int] = ..., forced_bos_token_id: Optional[int] = ..., forced_eos_token_id: Optional[int] = ..., length_penalty: Optional[float] = ..., early_stopping: Optional[bool] = ..., trace: bool = ..., params: Optional[Dict[str, jax_xla.DeviceArray]] = ..., **model_kwargs):
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        and, multinomial sampling.

        Apart from :obj:`input_ids`, all the arguments below will default to the value of the attribute of the same
        name inside the :class:`~transformers.PretrainedConfig` of the model. The default values indicated are the
        default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`jax_xla.DeviceArray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            temperature (:obj:`float`, `optional`, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or
                higher are kept for generation.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            trace (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to trace generation. Setting ``trace=False`` should only be used for debugging and will lead to
                a considerably slower runtime.
            params (:obj:`Dict[str, jax_xla.DeviceArray]`, `optional`):
                Optionally the model parameters can be passed. Can be useful for parallelized generation.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.

        Return:
            :class:`~transformers.file_utils.ModelOutput`.

        Examples::
            >>> from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = FlaxAutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> input_context = "The dog"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="jax").input_ids
            >>> # generate candidates using sampling
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, top_k=30, do_sample=True)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        ...
    


