

from torch import nn

from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_distilbert import DistilBertConfig

"""
 PyTorch DistilBERT model adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM) and in
 part from HuggingFace PyTorch version of Google AI Bert model (https://github.com/google-research/bert)
"""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
def create_sinusoidal_embeddings(n_pos, dim, out): # -> None:
    ...

class Embeddings(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids): # -> Any:
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.

        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        ...
    


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, query, key, value, mask, head_mask=..., output_attentions=...): # -> tuple[Any, Unknown | Any] | tuple[Any]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        ...
    


class FFN(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input): # -> Tensor:
        ...
    
    def ff_chunk(self, input): # -> Any:
        ...
    


class TransformerBlock(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, x, attn_mask=..., head_mask=..., output_attentions=...): # -> tuple[Any | Unbound, Any] | tuple[Any]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        ...
    


class Transformer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, x, attn_mask=..., head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        ...
    


class DistilBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DistilBertConfig
    load_tf_weights = ...
    base_model_prefix = ...


DISTILBERT_START_DOCSTRING = ...
DISTILBERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top.", DISTILBERT_START_DOCSTRING)
class DistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("""DistilBert Model with a `masked language modeling` head on top. """, DISTILBERT_START_DOCSTRING)
class DistilBertForMaskedLM(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | MaskedLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        """
        ...
    


@add_start_docstrings("""
    DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, DISTILBERT_START_DOCSTRING)
class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""
    DistilBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, DISTILBERT_START_DOCSTRING)
class DistilBertForQuestionAnswering(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., start_positions=..., end_positions=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        ...
    


@add_start_docstrings("""
    DistilBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """, DISTILBERT_START_DOCSTRING)
class DistilBertForTokenClassification(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=_CHECKPOINT_FOR_DOC, output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | TokenClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    


@add_start_docstrings("""
    DistilBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """, DISTILBERT_START_DOCSTRING)
class DistilBertForMultipleChoice(DistilBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @replace_return_docstrings(output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)

        Returns:

        Examples::

            >>> from transformers import DistilBertTokenizer, DistilBertForMultipleChoice
            >>> import torch

            >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            >>> model = DistilBertForMultipleChoice.from_pretrained('distilbert-base-cased')

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> choice0 = "It is eaten with a fork and a knife."
            >>> choice1 = "It is eaten while held in the hand."
            >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

            >>> encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors='pt', padding=True)
            >>> outputs = model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels) # batch size is 1

            >>> # the linear classifier still needs to be trained
            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        ...
    


