

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import MultipleChoiceModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from .configuration_visual_bert import VisualBertConfig

""" PyTorch VisualBERT model. """
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class VisualBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings and visual embeddings."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=..., visual_embeds=..., visual_token_type_ids=..., image_text_alignment=...):
        ...
    


class VisualBertSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...): # -> tuple[Tensor, Unknown | Any] | tuple[Tensor]:
        ...
    


class VisualBertSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class VisualBertAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...): # -> Any:
        ...
    


class VisualBertIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class VisualBertOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class VisualBertLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=...): # -> Any:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class VisualBertEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class VisualBertPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class VisualBertPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class VisualBertLMPredictionHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class VisualBertPreTrainingHeads(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, sequence_output, pooled_output): # -> tuple[Any, Any]:
        ...
    


class VisualBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = VisualBertConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...


@dataclass
class VisualBertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.VisualBertForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the sentence-image prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the sentence-image prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    prediction_logits: torch.FloatTensor = ...
    seq_relationship_logits: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


VISUAL_BERT_START_DOCSTRING = ...
VISUAL_BERT_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare VisualBert Model transformer outputting raw hidden-states without any specific head on top.", VISUAL_BERT_START_DOCSTRING)
class VisualBertModel(VisualBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., visual_embeds=..., visual_attention_mask=..., visual_token_type_ids=..., image_text_alignment=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""

        Returns:

        Example::

            >>> # Assumption: `get_visual_embeddings(image)` gets the visual embeddings of the image.
            >>> from transformers import BertTokenizer, VisualBertModel
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')

            >>> inputs = tokenizer("The capital of France is Paris.", return_tensors="pt")
            >>> visual_embeds = get_visual_embeddings(image).unsqueeze(0)
            >>> visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long) #example
            >>> visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

            >>> inputs.update({{
            ...     "visual_embeds": visual_embeds,
            ...     "visual_token_type_ids": visual_token_type_ids,
            ...     "visual_attention_mask": visual_attention_mask
            ... }})

            >>> outputs = model(**inputs)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        ...
    


@add_start_docstrings("""
    VisualBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `sentence-image prediction (classification)` head.
    """, VISUAL_BERT_START_DOCSTRING)
class VisualBertForPreTraining(VisualBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=VisualBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., visual_embeds=..., visual_attention_mask=..., visual_token_type_ids=..., image_text_alignment=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., sentence_image_labels=...):
        r"""
            labels (:obj:`torch.LongTensor` of shape ``(batch_size, total_sequence_length)``, `optional`):
                Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
                config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
                (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
            sentence_image_labels (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
                Labels for computing the sentence-image prediction (classification) loss. Input should be a sequence
                pair (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

                - 0 indicates sequence B is a matching pair of sequence A for the given image,
                - 1 indicates sequence B is a random sequence w.r.t A for the given image.

        Returns:

        Example::

            >>> # Assumption: `get_visual_embeddings(image)` gets the visual embeddings of the image in the batch.
            >>> from transformers import BertTokenizer, VisualBertForPreTraining

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = VisualBertForPreTraining.from_pretrained('uclanlp/visualbert-vqa-coco-pre')

            >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="pt")
            >>> visual_embeds = get_visual_embeddings(image).unsqueeze(0)
            >>> visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long) #example
            >>> visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

            >>> inputs.update({{
            ...     "visual_embeds": visual_embeds,
            ...     "visual_token_type_ids": visual_token_type_ids,
            ...     "visual_attention_mask": visual_attention_mask
            ... }})
            >>> max_length  = inputs["input_ids"].shape[-1]+visual_embeds.shape[-2]
            >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt", padding="max_length", max_length=max_length)["input_ids"]
            >>> sentence_image_labels = torch.tensor(1).unsqueeze(0) # Batch_size


            >>> outputs = model(**inputs, labels=labels, sentence_image_labels=sentence_image_labels)
            >>> loss = outputs.loss
            >>> prediction_logits = outputs.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits
        """
        ...
    


@add_start_docstrings("""
    VisualBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for VCR tasks.
    """, VISUAL_BERT_START_DOCSTRING)
class VisualBertForMultipleChoice(VisualBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @replace_return_docstrings(output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., visual_embeds=..., visual_attention_mask=..., visual_token_type_ids=..., image_text_alignment=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=...):
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
                num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors.
                (See :obj:`input_ids` above)

        Returns:

        Example::

            >>> from transformers import BertTokenizer, VisualBertForMultipleChoice
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = VisualBertForMultipleChoice.from_pretrained('uclanlp/visualbert-vcr')

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> choice0 = "It is eaten with a fork and a knife."
            >>> choice1 = "It is eaten while held in the hand."

            >>> visual_embeds = get_visual_embeddings(image)
            >>> # (batch_size, num_choices, visual_seq_length, visual_embedding_dim)
            >>> visual_embeds = visual_embeds.expand(1, 2, *visual_embeds.shape)
            >>> visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
            >>> visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

            >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

            >>> encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='pt', padding=True)
            >>> # batch size is 1
            >>> inputs_dict = {{k: v.unsqueeze(0) for k,v in encoding.items()}}
            >>> inputs_dict.update({{
            ... visual_embeds=visual_embeds,
            ... visual_attention_mask=visual_attention_mask,
            ... visual_token_type_ids=visual_token_type_ids,
            ... labels=labels
            ... }})
            >>> outputs = model(**inputs_dict)

            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        ...
    


@add_start_docstrings("""
    VisualBert Model with a classification/regression head on top (a dropout and a linear layer on top of the pooled
    output) for VQA.
    """, VISUAL_BERT_START_DOCSTRING)
class VisualBertForQuestionAnswering(VisualBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., visual_embeds=..., visual_attention_mask=..., visual_token_type_ids=..., image_text_alignment=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=...): # -> Any | SequenceClassifierOutput:
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, total_sequence_length)`, `optional`):
                Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
                config.num_labels - 1]`. A KLDivLoss is computed between the labels and the returned logits.

        Returns:

        Example::

            >>> # Assumption: `get_visual_embeddings(image)` gets the visual embeddings of the image in the batch.
            >>> from transformers import BertTokenizer, VisualBertForQuestionAnswering
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = VisualBertForQuestionAnswering.from_pretrained('uclanlp/visualbert-vqa')

            >>> text = "Who is eating the apple?"
            >>> inputs = tokenizer(text, return_tensors='pt')
            >>> visual_embeds = get_visual_embeddings(image).unsqueeze(0)
            >>> visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long) #example
            >>> visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

            >>> inputs.update({{
            ...     "visual_embeds": visual_embeds,
            ...     "visual_token_type_ids": visual_token_type_ids,
            ...     "visual_attention_mask": visual_attention_mask
            ... }})

            >>> labels = torch.tensor([[0.0,1.0]]).unsqueeze(0)  # Batch size 1, Num labels 2

            >>> outputs = model(**inputs, labels=labels)
            >>> loss = outputs.loss
            >>> scores = outputs.logits
        """
        ...
    


@add_start_docstrings("""
    VisualBert Model with a sequence classification head on top (a dropout and a linear layer on top of the pooled
    output) for Visual Reasoning e.g. for NLVR task.
    """, VISUAL_BERT_START_DOCSTRING)
class VisualBertForVisualReasoning(VisualBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., visual_embeds=..., visual_attention_mask=..., visual_token_type_ids=..., image_text_alignment=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=...): # -> Any | SequenceClassifierOutput:
        r"""
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
                config.num_labels - 1]`. A classification loss is computed (Cross-Entropy) against these labels.

        Returns:

        Example::

            >>> # Assumption: `get_visual_embeddings(image)` gets the visual embeddings of the image in the batch.
            >>> from transformers import BertTokenizer, VisualBertForVisualReasoning
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = VisualBertForVisualReasoning.from_pretrained('uclanlp/visualbert-nlvr2')

            >>> text = "Who is eating the apple?"
            >>> inputs = tokenizer(text, return_tensors='pt')
            >>> visual_embeds = get_visual_embeddings(image).unsqueeze(0)
            >>> visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long) #example
            >>> visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

            >>> inputs.update({{
            ...     "visual_embeds": visual_embeds,
            ...     "visual_token_type_ids": visual_token_type_ids,
            ...     "visual_attention_mask": visual_attention_mask
            ... }})

            >>> labels = torch.tensor(1).unsqueeze(0)  # Batch size 1, Num choices 2

            >>> outputs = model(**inputs, labels=labels)
            >>> loss = outputs.loss
            >>> scores = outputs.logits
        """
        ...
    


class VisualBertRegionToPhraseAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, query, key, attention_mask):
        ...
    


@add_start_docstrings("""
    VisualBert Model with a Masked Language Modeling head and an attention layer on top for Region-to-Phrase Alignment
    e.g. for Flickr30 Entities task.
    """, VISUAL_BERT_START_DOCSTRING)
class VisualBertForRegionToPhraseAlignment(VisualBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(VISUAL_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., visual_embeds=..., visual_attention_mask=..., visual_token_type_ids=..., image_text_alignment=..., output_attentions=..., output_hidden_states=..., return_dict=..., region_to_phrase_position=..., labels=...): # -> Any | SequenceClassifierOutput:
        r"""
            region_to_phrase_position (:obj:`torch.LongTensor` of shape ``(batch_size, total_sequence_length)``, `optional`):
                The positions depicting the position of the image embedding corresponding to the textual tokens.

            labels (:obj:`torch.LongTensor` of shape ``(batch_size, total_sequence_length, visual_sequence_length)``, `optional`):
                Labels for computing the masked language modeling loss. KLDivLoss is computed against these labels and
                the outputs from the attention layer.

        Returns:

        Example::

            >>> # Assumption: `get_visual_embeddings(image)` gets the visual embeddings of the image in the batch.
            >>> from transformers import BertTokenizer, VisualBertForRegionToPhraseAlignment
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = VisualBertForRegionToPhraseAlignment.from_pretrained('uclanlp/visualbert-vqa-coco-pre')

            >>> text = "Who is eating the apple?"
            >>> inputs = tokenizer(text, return_tensors='pt')
            >>> visual_embeds = get_visual_embeddings(image).unsqueeze(0)
            >>> visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long) #example
            >>> visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
            >>> region_to_phrase_position = torch.ones((1, inputs["input_ids"].shape[-1]+visual_embeds.shape[-2]))

            >>> inputs.update({{
            ...     "region_to_phrase_position": region_to_phrase_position,
            ...     "visual_embeds": visual_embeds,
            ...     "visual_token_type_ids": visual_token_type_ids,
            ...     "visual_attention_mask": visual_attention_mask
            ... }})

            >>> labels = torch.ones((1, inputs["input_ids"].shape[-1]+visual_embeds.shape[-2], visual_embeds.shape[-2])) # Batch size 1

            >>> outputs = model(**inputs, labels=labels)
            >>> loss = outputs.loss
            >>> scores = outputs.logits
        """
        ...
    


