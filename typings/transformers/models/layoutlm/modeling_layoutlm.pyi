

from torch import nn

from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_layoutlm import LayoutLMConfig

""" PyTorch LayoutLM model. """
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = ...
LayoutLMLayerNorm = nn.LayerNorm
class LayoutLMEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., bbox=..., token_type_ids=..., position_ids=..., inputs_embeds=...):
        ...
    


class LayoutLMSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_value=..., output_attentions=...):
        ...
    


class LayoutLMSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class LayoutLMAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_value=..., output_attentions=...): # -> Any:
        ...
    


class LayoutLMIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class LayoutLMOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class LayoutLMLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_value=..., output_attentions=...): # -> Any:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class LayoutLMEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class LayoutLMPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class LayoutLMPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class LayoutLMLMPredictionHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class LayoutLMOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, sequence_output): # -> Any:
        ...
    


class LayoutLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LayoutLMConfig
    pretrained_model_archive_map = ...
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...


LAYOUTLM_START_DOCSTRING = ...
LAYOUTLM_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top.", LAYOUTLM_START_DOCSTRING)
class LayoutLMModel(LayoutLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., bbox=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., encoder_hidden_states=..., encoder_attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, LayoutLMModel
            >>> import torch

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMModel.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        ...
    


@add_start_docstrings("""LayoutLM Model with a `language modeling` head on top. """, LAYOUTLM_START_DOCSTRING)
class LayoutLMForMaskedLM(LayoutLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., bbox=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., encoder_hidden_states=..., encoder_attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | MaskedLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, LayoutLMForMaskedLM
            >>> import torch

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMForMaskedLM.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "[MASK]"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])

            >>> labels = tokenizer("Hello world", return_tensors="pt")["input_ids"]

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=labels)

            >>> loss = outputs.loss
        """
        ...
    


@add_start_docstrings("""
    LayoutLM Model with a sequence classification head on top (a linear layer on top of the pooled output) e.g. for
    document image classification tasks such as the `RVL-CDIP <https://www.cs.cmu.edu/~aharley/rvl-cdip/>`__ dataset.
    """, LAYOUTLM_START_DOCSTRING)
class LayoutLMForSequenceClassification(LayoutLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., bbox=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | SequenceClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, LayoutLMForSequenceClassification
            >>> import torch

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMForSequenceClassification.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])
            >>> sequence_label = torch.tensor([1])

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=sequence_label)

            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        ...
    


@add_start_docstrings("""
    LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    sequence labeling (information extraction) tasks such as the `FUNSD <https://guillaumejaume.github.io/FUNSD/>`__
    dataset and the `SROIE <https://rrc.cvc.uab.es/?ch=13>`__ dataset.
    """, LAYOUTLM_START_DOCSTRING)
class LayoutLMForTokenClassification(LayoutLMPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., bbox=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | TokenClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.

        Returns:

        Examples::

            >>> from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
            >>> import torch

            >>> tokenizer = LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased')
            >>> model = LayoutLMForTokenClassification.from_pretrained('microsoft/layoutlm-base-uncased')

            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            >>> encoding = tokenizer(' '.join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])
            >>> token_labels = torch.tensor([1,1,0,0]).unsqueeze(0) # batch size of 1

            >>> outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
            ...                 labels=token_labels)

            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        ...
    


