

from torch import nn

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_squeezebert import SqueezeBertConfig

""" PyTorch SqueezeBert model. """
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class SqueezeBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...): # -> Any:
        ...
    


class MatMulWrapper(nn.Module):
    """
    Wrapper for torch.matmul(). This makes flop-counting easier to implement. Note that if you directly call
    torch.matmul() in your code, the flop counter will typically ignore the flops of the matmul.
    """
    def __init__(self) -> None:
        ...
    
    def forward(self, mat1, mat2): # -> Tensor:
        """

        :param inputs: two torch tensors :return: matmul of these tensors

        Here are the typical dimensions found in BERT (the B is optional) mat1.shape: [B, <optional extra dims>, M, K]
        mat2.shape: [B, <optional extra dims>, K, N] output shape: [B, <optional extra dims>, M, N]
        """
        ...
    


class SqueezeBertLayerNorm(nn.LayerNorm):
    """
    This is a nn.LayerNorm subclass that accepts NCW data layout and performs normalization in the C dimension.

    N = batch C = channels W = sequence length
    """
    def __init__(self, hidden_size, eps=...) -> None:
        ...
    
    def forward(self, x): # -> Tensor:
        ...
    


class ConvDropoutLayerNorm(nn.Module):
    """
    ConvDropoutLayerNorm: Conv, Dropout, LayerNorm
    """
    def __init__(self, cin, cout, groups, dropout_prob) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class ConvActivation(nn.Module):
    """
    ConvActivation: Conv, Activation
    """
    def __init__(self, cin, cout, groups, act) -> None:
        ...
    
    def forward(self, x): # -> Any | Tensor:
        ...
    


class SqueezeBertSelfAttention(nn.Module):
    def __init__(self, config, cin, q_groups=..., k_groups=..., v_groups=...) -> None:
        """
        config = used for some things; ignored for others (work in progress...) cin = input channels = output channels
        groups = number of groups to use in conv1d layers
        """
        ...
    
    def transpose_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, W, C2] where C1 is the head index, and C2 is one head's contents
        """
        ...
    
    def transpose_key_for_scores(self, x):
        """
        - input: [N, C, W]
        - output: [N, C1, C2, W] where C1 is the head index, and C2 is one head's contents
        """
        ...
    
    def transpose_output(self, x):
        """
        - input: [N, C1, W, C2]
        - output: [N, C, W]
        """
        ...
    
    def forward(self, hidden_states, attention_mask, output_attentions): # -> dict[str, Any]:
        """
        expects hidden_states in [N, C, W] data layout.

        The attention_mask data layout is [N, W], and it does not need to be transposed.
        """
        ...
    


class SqueezeBertModule(nn.Module):
    def __init__(self, config) -> None:
        """
        - hidden_size = input chans = output chans for Q, K, V (they are all the same ... for now) = output chans for
          the module
        - intermediate_size = output chans for intermediate layer
        - groups = number of groups for all layers in the BertModule. (eventually we could change the interface to
          allow different groups for different layers)
        """
        ...
    
    def forward(self, hidden_states, attention_mask, output_attentions): # -> dict[str, Any]:
        ...
    


class SqueezeBertEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class SqueezeBertPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class SqueezeBertPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class SqueezeBertLMPredictionHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class SqueezeBertOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, sequence_output): # -> Any:
        ...
    


class SqueezeBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = SqueezeBertConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...


SQUEEZEBERT_START_DOCSTRING = ...
SQUEEZEBERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare SqueezeBERT Model transformer outputting raw hidden-states without any specific head on top.", SQUEEZEBERT_START_DOCSTRING)
class SqueezeBertModel(SqueezeBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("""SqueezeBERT Model with a `language modeling` head on top. """, SQUEEZEBERT_START_DOCSTRING)
class SqueezeBertForMaskedLM(SqueezeBertPreTrainedModel):
    _keys_to_ignore_on_load_missing = ...
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | MaskedLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        ...
    


@add_start_docstrings("""
    SqueezeBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, SQUEEZEBERT_START_DOCSTRING)
class SqueezeBertForSequenceClassification(SqueezeBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""
    SqueezeBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """, SQUEEZEBERT_START_DOCSTRING)
class SqueezeBertForMultipleChoice(SqueezeBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("(batch_size, num_choices, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where `num_choices` is the size of the second dimension of the input tensors. (see
            `input_ids` above)
        """
        ...
    


@add_start_docstrings("""
    SqueezeBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """, SQUEEZEBERT_START_DOCSTRING)
class SqueezeBertForTokenClassification(SqueezeBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | TokenClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    


@add_start_docstrings("""
     SqueezeBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
     linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
     """, SQUEEZEBERT_START_DOCSTRING)
class SqueezeBertForQuestionAnswering(SqueezeBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(SQUEEZEBERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., start_positions=..., end_positions=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        ...
    


