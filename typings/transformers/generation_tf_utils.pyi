

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorflow as tf

from .file_utils import ModelOutput

logger = ...
@dataclass
class TFGreedySearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using greedy search.


    Args:
        sequences (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(tf.Tensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length-input_ids.shape[-1],)`-shaped tuple of :obj:`tf.Tensor` with
            each tensor of shape :obj:`(batch_size, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size, generated_length, hidden_size)`.
    """
    sequences: tf.Tensor = ...
    scores: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = ...
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = ...


@dataclass
class TFGreedySearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using greedy search. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(tf.Tensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length-1,)`-shaped tuple of :obj:`tf.Tensor` with each tensor of shape
            :obj:`(batch_size, config.vocab_size)`).
        encoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer of the decoder) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size, generated_length, hidden_size)`.
    """
    sequences: tf.Tensor = ...
    scores: Optional[Tuple[tf.Tensor]] = ...
    encoder_attentions: Optional[Tuple[tf.Tensor]] = ...
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = ...
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = ...
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = ...


@dataclass
class TFSampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using sampling.


    Args:
        sequences (:obj:`tf.Tensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(tf.Tensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length-input_ids.shape[-1],)`-shaped tuple of :obj:`tf.Tensor` with
            each tensor of shape :obj:`(batch_size*num_return_sequences, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(num_return_sequences*batch_size, num_heads, generated_length,
            sequence_length)`.
        hidden_states (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(num_return_sequences*batch_size, generated_length, hidden_size)`.
    """
    sequences: tf.Tensor = ...
    scores: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = ...
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = ...


@dataclass
class TFSampleEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using sampling. Hidden states and attention weights of
    the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)


    Args:
        sequences (:obj:`tf.Tensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        scores (:obj:`tuple(tf.Tensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. :obj:`(max_length-1,)`-shaped tuple of :obj:`tf.Tensor` with each tensor of shape
            :obj:`(batch_size*num_return_sequences, config.vocab_size)`).
        encoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer of the decoder) of shape
            :obj:`(batch_size*num_return_sequences, num_heads, sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size*num_return_sequences, generated_length, hidden_size)`.
    """
    sequences: tf.Tensor = ...
    scores: Optional[Tuple[tf.Tensor]] = ...
    encoder_attentions: Optional[Tuple[tf.Tensor]] = ...
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = ...
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = ...
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = ...


@dataclass
class TFBeamSearchDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam search.

    Args:
        sequences (:obj:`tf.Tensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size*num_return_sequences)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(tf.Tensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length-input_ids.shape[-1],)`-shaped tuple of :obj:`tf.Tensor` with each tensor of shape
            :obj:`(batch_size*num_beams*num_return_sequences, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        hidden_states (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, generated_length,
            hidden_size)`.
    """
    sequences: tf.Tensor = ...
    sequences_scores: Optional[tf.Tensor] = ...
    scores: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = ...
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = ...


@dataclass
class TFBeamSearchEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights
    of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the encoder_hidden_states
    attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (:obj:`tf.Tensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size*num_return_sequences)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(tf.Tensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length-1,)`-shaped tuple of :obj:`tf.Tensor` with each tensor of shape
            :obj:`(batch_size*num_beams, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
        encoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer of the decoder) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size*num_beams*num_return_sequences, generated_length,
            hidden_size)`.
    """
    sequences: tf.Tensor = ...
    sequences_scores: Optional[tf.Tensor] = ...
    scores: Optional[Tuple[tf.Tensor]] = ...
    encoder_attentions: Optional[Tuple[tf.Tensor]] = ...
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = ...
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = ...
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = ...


@dataclass
class TFBeamSampleDecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using beam sample.

    Args:
        sequences (:obj:`tf.Tensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size * num_return_sequence)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(tf.Tensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length-input_ids.shape[-1],)`-shaped tuple of :obj:`tf.Tensor` with each tensor of shape
            :obj:`(batch_size*num_beams*num_return_sequences, config.vocab_size)`).
        attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        hidden_states (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size*num_beams, generated_length, hidden_size)`.
    """
    sequences: tf.Tensor = ...
    sequences_scores: Optional[tf.Tensor] = ...
    scores: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[Tuple[tf.Tensor]]] = ...
    hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = ...


@dataclass
class TFBeamSampleEncoderDecoderOutput(ModelOutput):
    """
    Base class for outputs of encoder-decoder generation models using beam sampling. Hidden states and attention
    weights of the decoder (respectively the encoder) can be accessed via the encoder_attentions and the
    encoder_hidden_states attributes (respectively the decoder_attentions and the decoder_hidden_states attributes)

    Args:
        sequences (:obj:`tf.Tensor` of shape :obj:`(batch_size*num_beams, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.
        sequences_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size * num_return_sequence)`, `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Final beam scores of the generated ``sequences``.
        scores (:obj:`tuple(tf.Tensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
            Processed beam scores for each vocabulary token at each generation step. Beam scores consisting of log
            softmax scores for each vocabulary token and sum of log softmax of previously generated tokens in this beam
            . :obj:`(max_length-1,)`-shaped tuple of :obj:`tf.Tensor` with each tensor of shape
            :obj:`(batch_size*num_beams, config.vocab_size)`).
        encoder_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer of the decoder) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size*num_beams, sequence_length, hidden_size)`.
        decoder_attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        cross_attentions (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (:obj:`tuple(tuple(tf.Tensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            :obj:`tf.Tensor` of shape :obj:`(batch_size*num_beams, generated_length, hidden_size)`.
    """
    sequences: tf.Tensor = ...
    sequences_scores: Optional[tf.Tensor] = ...
    scores: Optional[Tuple[tf.Tensor]] = ...
    encoder_attentions: Optional[Tuple[tf.Tensor]] = ...
    encoder_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    decoder_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = ...
    cross_attentions: Optional[Tuple[Tuple[tf.Tensor]]] = ...
    decoder_hidden_states: Optional[Tuple[Tuple[tf.Tensor]]] = ...


TFGreedySearchOutput = Union[TFGreedySearchEncoderDecoderOutput, TFGreedySearchDecoderOnlyOutput]
TFSampleOutput = Union[TFSampleEncoderDecoderOutput, TFSampleDecoderOnlyOutput]
TFBeamSearchOutput = Union[TFBeamSearchEncoderDecoderOutput, TFBeamSearchDecoderOnlyOutput]
TFBeamSampleOutput = Union[TFBeamSampleEncoderDecoderOutput, TFBeamSampleDecoderOnlyOutput]
class TFGenerationMixin:
    """
    A class containing all of the functions supporting generation, to be used as a mixin in
    :class:`~transformers.TFPreTrainedModel`.
    """
    def prepare_inputs_for_generation(self, inputs, **kwargs):
        """
        Implement in subclasses of :class:`~transformers.TFPreTrainedModel` for custom behavior to prepare inputs in
        the generate method.
        """
        ...
    
    def generate(self, input_ids=..., max_length=..., min_length=..., do_sample=..., early_stopping=..., num_beams=..., temperature=..., top_k=..., top_p=..., repetition_penalty=..., bad_words_ids=..., bos_token_id=..., pad_token_id=..., eos_token_id=..., length_penalty=..., no_repeat_ngram_size=..., num_return_sequences=..., attention_mask=..., decoder_start_token_id=..., use_cache=..., output_scores=..., output_attentions=..., output_hidden_states=..., return_dict_in_generate=..., forced_bos_token_id=..., forced_eos_token_id=..., **model_kwargs) -> Union[TFGreedySearchOutput, TFSampleOutput, TFBeamSearchOutput, TFBeamSampleOutput, tf.Tensor]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from `Facebook's XLM beam search code
        <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`tf.Tensor` of :obj:`dtype=tf.int32` and shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`tf.Tensor` of shape :obj:`(1,)`.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.

                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(:obj:`List[int]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (:obj:`tf.Tensor` of :obj:`dtype=tf.int32` and shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens.

                If not provided, will default to a tensor the same shape as :obj:`input_ids` that masks the pad token.

                `What are attention masks? <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            forced_bos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the first generated token after the :obj:`decoder_start_token_id`.
                Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token
                needs to be the target language token.
            forced_eos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the last generated token when :obj:`max_length` is reached.
            model_specific_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.

        Return:
            :class:`~transformers.file_utils.ModelOutput` or :obj:`tf.Tensor`: A
            :class:`~transformers.file_utils.ModelOutput` (if ``return_dict_in_generate=True`` or when
            ``config.return_dict_in_generate=True``) or a :obj:`tf.Tensor`.

                If the model is `not` an encoder-decoder model (``model.config.is_encoder_decoder=False``), the
                possible :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.TFGreedySearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.TFSampleDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.TFBeamSearchDecoderOnlyOutput`,
                    - :class:`~transformers.generation_utils.TFBeamSampleDecoderOnlyOutput`

                If the model is an encoder-decoder model (``model.config.is_encoder_decoder=True``), the possible
                :class:`~transformers.file_utils.ModelOutput` types are:

                    - :class:`~transformers.generation_utils.TFGreedySearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.TFSampleEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.TFBeamSearchEncoderDecoderOutput`,
                    - :class:`~transformers.generation_utils.TFBeamSampleEncoderDecoderOutput`

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from huggingface.co and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print(f'Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}')

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from huggingface.co and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print(f'Generated {i}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from huggingface.co and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True)  # generate 3 candidates using sampling
            for i in range(3): #  3 output sequences were generated
                print(f'Generated {i}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from huggingface.co and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print(f'Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}')

            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from huggingface.co and cache.
            input_context = 'My cute dog'
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """
        ...
    
    def adjust_logits_during_generation(self, logits, cur_len, max_length, forced_bos_token_id, forced_eos_token_id, **kwargs):
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to adjust the logits in
        the generate method.
        """
        ...
    


def calc_banned_ngram_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len):
    ...

def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    ...

def tf_top_k_top_p_filtering(logits, top_k=..., top_p=..., filter_value=..., min_tokens_to_keep=...):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    ...

def scatter_values_on_batch_indices(values, batch_indices):
    ...

def set_tensor_by_indices_to_value(tensor, indices, value):
    ...

def sample_without_replacement(logits, num_samples):
    """
    categorical sampling without replacement is currently not implemented the gumbel-max trick will do for now see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    ...

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    ...

class BeamHypotheses:
    def __init__(self, num_beams, max_length, length_penalty, early_stopping) -> None:
        """
        Initialize n-best list of hypotheses.
        """
        ...
    
    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        ...
    
    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        ...
    
    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        ...
    


