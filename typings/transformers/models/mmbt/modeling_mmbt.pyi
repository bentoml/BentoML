

from torch import nn

from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import ModuleUtilsMixin

"""PyTorch MMBT model. """
logger = ...
_CONFIG_FOR_DOC = ...
class ModalEmbeddings(nn.Module):
    """Generic Modal Embeddings which takes in an encoder, and a transformer embedding."""
    def __init__(self, config, encoder, embeddings) -> None:
        ...
    
    def forward(self, input_modal, start_token=..., end_token=..., position_ids=..., token_type_ids=...): # -> Any:
        ...
    


MMBT_START_DOCSTRING = ...
MMBT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare MMBT Model outputting raw hidden-states without any specific head on top.", MMBT_START_DOCSTRING)
class MMBTModel(nn.Module, ModuleUtilsMixin):
    def __init__(self, config, transformer, encoder) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(MMBT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_modal, input_ids=..., modal_start_tokens=..., modal_end_tokens=..., attention_mask=..., token_type_ids=..., modal_token_type_ids=..., position_ids=..., modal_position_ids=..., head_mask=..., inputs_embeds=..., encoder_hidden_states=..., encoder_attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Returns:

        Examples::

            # For example purposes. Not runnable.
            transformer = BertModel.from_pretrained('bert-base-uncased')
            encoder = ImageEncoder(args)
            mmbt = MMBTModel(config, transformer, encoder)
        """
        ...
    
    def get_input_embeddings(self): # -> Tensor | Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    


@add_start_docstrings("""
    MMBT Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    """, MMBT_START_DOCSTRING, MMBT_INPUTS_DOCSTRING)
class MMBTForClassification(nn.Module):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss. Indices should be in ``[0, ...,
            config.num_labels - 1]``. If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Returns: `Tuple` comprising various elements depending on the configuration (config) and inputs: **loss**:
    (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``: Classification (or
    regression if config.num_labels==1) loss. **logits**: ``torch.FloatTensor`` of shape ``(batch_size,
    config.num_labels)`` Classification (or regression if config.num_labels==1) scores (before SoftMax).
    **hidden_states**: (`optional`, returned when ``output_hidden_states=True``) list of ``torch.FloatTensor`` (one for
    the output of each layer + the output of the embeddings) of shape ``(batch_size, sequence_length, hidden_size)``:
    Hidden-states of the model at the output of each layer plus the initial embedding outputs. **attentions**:
    (`optional`, returned when ``output_attentions=True``) list of ``torch.FloatTensor`` (one for each layer) of shape
    ``(batch_size, num_heads, sequence_length, sequence_length)``: Attentions weights after the attention softmax, used
    to compute the weighted average in the self-attention heads.

    Examples::

        # For example purposes. Not runnable.
        transformer = BertModel.from_pretrained('bert-base-uncased')
        encoder = ImageEncoder(args)
        model = MMBTForClassification(config, transformer, encoder)
        outputs = model(input_modal, input_ids, labels=labels)
        loss, logits = outputs[:2]
    """
    def __init__(self, config, transformer, encoder) -> None:
        ...
    
    def forward(self, input_modal, input_ids=..., modal_start_tokens=..., modal_end_tokens=..., attention_mask=..., token_type_ids=..., modal_token_type_ids=..., position_ids=..., modal_position_ids=..., head_mask=..., inputs_embeds=..., labels=..., return_dict=...): # -> Any | SequenceClassifierOutput:
        ...
    


