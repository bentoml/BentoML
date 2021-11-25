

from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf

from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMultipleChoiceLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    keras_serializable,
)
from .configuration_xlnet import XLNetConfig

"""
 TF 2.0 XLNet model.
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class TFXLNetRelativeAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def prune_heads(self, heads):
        ...
    
    def rel_shift(self, x, klen=...):
        """perform relative shift to form the relative attention score."""
        ...
    
    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask, head_mask, output_attentions, training=...): # -> tuple[Unknown, Unknown]:
        """Core relative positional attention operations."""
        ...
    
    def post_attention(self, h, attn_vec, residual=..., training=...):
        """Post-attention processing."""
        ...
    
    def call(self, h, g, attn_mask_h, attn_mask_g, r, seg_mat, mems, target_mapping, head_mask, output_attentions, training=...):
        ...
    


class TFXLNetFeedForward(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, inp, training=...):
        ...
    


class TFXLNetLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, output_h, output_g, non_tgt_mask, attn_mask, pos_emb, seg_mat, mems, target_mapping, head_mask, output_attentions, training=...):
        ...
    


class TFXLNetLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def get_output_embeddings(self):
        ...
    
    def set_output_embeddings(self, value): # -> None:
        ...
    
    def get_bias(self): # -> dict[str, Unknown]:
        ...
    
    def set_bias(self, value): # -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


@keras_serializable
class TFXLNetMainLayer(tf.keras.layers.Layer):
    config_class = XLNetConfig
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def get_input_embeddings(self): # -> TFSharedEmbeddings:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def create_mask(self, qlen, mlen):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.

        Args:
            qlen: TODO Lysandre didn't fill
            mlen: TODO Lysandre didn't fill

        ::

                  same_length=False:      same_length=True:
                  <mlen > <  qlen >       <mlen > <  qlen >
               ^ [0 0 0 0 0 1 1 1 1]     [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1]     [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1]     [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1]     [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0]     [1 1 1 1 0 0 0 0 0]

        """
        ...
    
    def cache_mem(self, curr_out, prev_mem):
        ...
    
    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=...):
        ...
    
    def relative_positional_encoding(self, qlen, klen, bsz=...):
        """create relative positional encoding."""
        ...
    
    def call(self, input_ids=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., token_type_ids=..., input_mask=..., head_mask=..., inputs_embeds=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    


class TFXLNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XLNetConfig
    base_model_prefix = ...


@dataclass
class TFXLNetModelOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFXLNetModel`.

    Args:
        last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            ``num_predict`` corresponds to ``target_mapping.shape[1]``. If ``target_mapping`` is ``None``, then
            ``num_predict`` corresponds to ``sequence_length``.
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: tf.Tensor = ...
    mems: Optional[List[tf.Tensor]] = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFXLNetLMHeadModelOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFXLNetLMHeadModel`.

    Args:
        loss (:obj:`tf.Tensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
            Language modeling loss (for next-token prediction).
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_predict, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            ``num_predict`` corresponds to ``target_mapping.shape[1]``. If ``target_mapping`` is ``None``, then
            ``num_predict`` corresponds to ``sequence_length``.
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[tf.Tensor] = ...
    logits: tf.Tensor = ...
    mems: Optional[List[tf.Tensor]] = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFXLNetForSequenceClassificationOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFXLNetForSequenceClassification`.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[tf.Tensor] = ...
    logits: tf.Tensor = ...
    mems: Optional[List[tf.Tensor]] = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFXLNetForTokenClassificationOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFXLNetForTokenClassificationOutput`.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[tf.Tensor] = ...
    logits: tf.Tensor = ...
    mems: Optional[List[tf.Tensor]] = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFXLNetForMultipleChoiceOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFXLNetForMultipleChoice`.

    Args:
        loss (:obj:`tf.Tensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices)`):
            `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

            Classification scores (before SoftMax).
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[tf.Tensor] = ...
    logits: tf.Tensor = ...
    mems: Optional[List[tf.Tensor]] = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFXLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFXLNetForQuestionAnsweringSimple`.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        mems (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see :obj:`mems` input) to speed up sequential decoding.
            The token ids which have their past given to this model should not be passed as :obj:`input_ids` as they
            have already been computed.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[tf.Tensor] = ...
    start_logits: tf.Tensor = ...
    end_logits: tf.Tensor = ...
    mems: Optional[List[tf.Tensor]] = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


XLNET_START_DOCSTRING = ...
XLNET_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare XLNet Model transformer outputting raw hidden-states without any specific head on top.", XLNET_START_DOCSTRING)
class TFXLNetModel(TFXLNetPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLNetModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., token_type_ids=..., input_mask=..., head_mask=..., inputs_embeds=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output): # -> TFXLNetModelOutput:
        ...
    


@add_start_docstrings("""
    XLNet Model with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """, XLNET_START_DOCSTRING)
class TFXLNetLMHeadModel(TFXLNetPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self): # -> TFXLNetLMHead:
        ...
    
    def get_prefix_bias_name(self):
        ...
    
    def prepare_inputs_for_generation(self, inputs, past, use_mems=..., **kwargs): # -> dict[str, Unknown | None]:
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFXLNetLMHeadModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., token_type_ids=..., input_mask=..., head_mask=..., inputs_embeds=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFXLNetLMHeadModelOutput:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.

        Return:

        Examples::

            >>> import tensorflow as tf
            >>> import numpy as np
            >>> from transformers import XLNetTokenizer, TFXLNetLMHeadModel

            >>> tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
            >>> model = TFXLNetLMHeadModel.from_pretrained('xlnet-large-cased')

            >>> # We show how to setup inputs to predict a next token using a bi-directional context.
            >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=True))[None, :]  # We will predict the masked token

            >>> perm_mask = np.zeros((1, input_ids.shape[1], input_ids.shape[1]))
            >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token

            >>> target_mapping = np.zeros((1, 1, input_ids.shape[1]))  # Shape [1, 1, seq_length] => let's predict one token
            >>> target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

            >>> outputs = model(input_ids, perm_mask=tf.constant(perm_mask, dtype=tf.float32), target_mapping=tf.constant(target_mapping, dtype=tf.float32))

            >>> next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

        """
        ...
    
    def serving_output(self, output): # -> TFXLNetLMHeadModelOutput:
        ...
    


@add_start_docstrings("""
    XLNet Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.
    """, XLNET_START_DOCSTRING)
class TFXLNetForSequenceClassification(TFXLNetPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLNetForSequenceClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., token_type_ids=..., input_mask=..., head_mask=..., inputs_embeds=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFXLNetForSequenceClassificationOutput:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
        """
        ...
    
    def serving_output(self, output): # -> TFXLNetForSequenceClassificationOutput:
        ...
    


@add_start_docstrings("""
    XLNET Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, XLNET_START_DOCSTRING)
class TFXLNetForMultipleChoice(TFXLNetPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @property
    def dummy_inputs(self): # -> dict[str, Unknown]:
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLNetForMultipleChoiceOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., token_type_ids=..., input_mask=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., head_mask=..., inputs_embeds=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),"token_type_ids": tf.TensorSpec((None, None, None), tf.int32, name="token_type_ids") }])
    def serving(self, inputs): # -> TFXLNetForMultipleChoiceOutput:
        ...
    
    def serving_output(self, output): # -> TFXLNetForMultipleChoiceOutput:
        ...
    


@add_start_docstrings("""
    XLNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, XLNET_START_DOCSTRING)
class TFXLNetForTokenClassification(TFXLNetPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLNetForTokenClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., token_type_ids=..., input_mask=..., head_mask=..., inputs_embeds=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs): # -> TFXLNetForTokenClassificationOutput:
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    
    def serving_output(self, output): # -> TFXLNetForTokenClassificationOutput:
        ...
    


@add_start_docstrings("""
    XLNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, XLNET_START_DOCSTRING)
class TFXLNetForQuestionAnsweringSimple(TFXLNetPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XLNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TFXLNetForQuestionAnsweringSimpleOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., mems=..., perm_mask=..., target_mapping=..., token_type_ids=..., input_mask=..., head_mask=..., inputs_embeds=..., use_mems=..., output_attentions=..., output_hidden_states=..., return_dict=..., start_positions=..., end_positions=..., training=..., **kwargs): # -> TFXLNetForQuestionAnsweringSimpleOutput:
        r"""
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        ...
    
    def serving_output(self, output): # -> TFXLNetForQuestionAnsweringSimpleOutput:
        ...
    


