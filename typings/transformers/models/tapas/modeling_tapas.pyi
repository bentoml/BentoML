

import enum
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scatter_available,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_tapas import TapasConfig

"""PyTorch TAPAS model. """
if is_scatter_available():
    ...
logger = ...
_CONFIG_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
_TOKENIZER_FOR_DOC = ...
TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST = ...
EPSILON_ZERO_DIVISION = ...
CLOSE_ENOUGH_TO_LOG_ZERO = ...
@dataclass
class TableQuestionAnsweringOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TapasForQuestionAnswering`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` (and possibly :obj:`answer`, :obj:`aggregation_labels`, :obj:`numeric_values` and :obj:`numeric_values_scale` are provided)):
            Total loss as the sum of the hierarchical cell selection log-likelihood loss and (optionally) the
            semi-supervised regression loss and (optionally) supervised loss for aggregations.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the cell selection head, for every token.
        logits_aggregation (:obj:`torch.FloatTensor`, `optional`, of shape :obj:`(batch_size, num_aggregation_labels)`):
            Prediction scores of the aggregation head, for every aggregation operator.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of
            each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    logits_aggregation: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


def load_tf_weights_in_tapas(model, config, tf_checkpoint_path):
    """
    Load tf checkpoints in a PyTorch model. This is an adaptation from load_tf_weights_in_bert

    - add cell selection and aggregation heads
    - take into account additional token type embedding layers
    """
    ...

class TapasEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings. Same as BertEmbeddings but with a number of
    additional token type embeddings to encode tabular structure.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...): # -> Any:
        ...
    


class TapasSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_value=..., output_attentions=...):
        ...
    


class TapasSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class TapasAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_value=..., output_attentions=...): # -> Any:
        ...
    


class TapasIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any | Tensor:
        ...
    


class TapasOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor): # -> Any:
        ...
    


class TapasLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_value=..., output_attentions=...): # -> Any:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class TapasEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class TapasPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class TapasPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = TapasConfig
    base_model_prefix = ...


TAPAS_START_DOCSTRING = ...
TAPAS_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Tapas Model transformer outputting raw hidden-states without any specific head on top.", TAPAS_START_DOCSTRING)
class TapasModel(TapasPreTrainedModel):
    """
    This class is a small change compared to :class:`~transformers.BertModel`, taking into account the additional token
    type ids.

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    """
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., encoder_hidden_states=..., encoder_attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Returns:

        Examples::

            >>> from transformers import TapasTokenizer, TapasModel
            >>> import pandas as pd

            >>> tokenizer = TapasTokenizer.from_pretrained('google/tapas-base')
            >>> model = TapasModel.from_pretrained('google/tapas-base')

            >>> data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
            ...         'Age': ["56", "45", "59"],
            ...         'Number of movies': ["87", "53", "69"]
            ... }
            >>> table = pd.DataFrame.from_dict(data)
            >>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

            >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        ...
    


@add_start_docstrings("""Tapas Model with a `language modeling` head on top. """, TAPAS_START_DOCSTRING)
class TapasForMaskedLM(TapasPreTrainedModel):
    config_class = TapasConfig
    base_model_prefix = ...
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, word_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., encoder_hidden_states=..., encoder_attention_mask=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs): # -> Any | MaskedLMOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import TapasTokenizer, TapasForMaskedLM
            >>> import pandas as pd

            >>> tokenizer = TapasTokenizer.from_pretrained('google/tapas-base')
            >>> model = TapasForMaskedLM.from_pretrained('google/tapas-base')

            >>> data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
            ...         'Age': ["56", "45", "59"],
            ...         'Number of movies': ["87", "53", "69"]
            ... }
            >>> table = pd.DataFrame.from_dict(data)

            >>> inputs = tokenizer(table=table, queries="How many [MASK] has George [MASK] played in?", return_tensors="pt")
            >>> labels = tokenizer(table=table, queries="How many movies has George Clooney played in?", return_tensors="pt")["input_ids"]

            >>> outputs = model(**inputs, labels=labels)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        ...
    


@add_start_docstrings("""
    Tapas Model with a cell selection head and optional aggregation head on top for question-answering tasks on tables
    (linear layers on top of the hidden-states output to compute `logits` and optional `logits_aggregation`), e.g. for
    SQA, WTQ or WikiSQL-supervised tasks.
    """, TAPAS_START_DOCSTRING)
class TapasForQuestionAnswering(TapasPreTrainedModel):
    def __init__(self, config: TapasConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TableQuestionAnsweringOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., table_mask=..., labels=..., aggregation_labels=..., float_answer=..., numeric_values=..., numeric_values_scale=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        table_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_length)`, `optional`):
            Mask for the table. Indicates which tokens belong to the table (1). Question tokens, table headers and
            padding are 0.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_length)`, `optional`):
            Labels per token for computing the hierarchical cell selection loss. This encodes the positions of the
            answer appearing in the table. Can be obtained using :class:`~transformers.TapasTokenizer`.

            - 1 for tokens that are **part of the answer**,
            - 0 for tokens that are **not part of the answer**.

        aggregation_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, )`, `optional`):
            Aggregation function index for every example in the batch for computing the aggregation loss. Indices
            should be in :obj:`[0, ..., config.num_aggregation_labels - 1]`. Only required in case of strong
            supervision for aggregation (WikiSQL-supervised).
        float_answer (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, )`, `optional`):
            Float answer for every example in the batch. Set to `float('nan')` for cell selection questions. Only
            required in case of weak supervision (WTQ) to calculate the aggregate mask and regression loss.
        numeric_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length)`, `optional`):
            Numeric values of every token, NaN for tokens which are not numeric values. Can be obtained using
            :class:`~transformers.TapasTokenizer`. Only required in case of weak supervision for aggregation (WTQ) to
            calculate the regression loss.
        numeric_values_scale (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length)`, `optional`):
            Scale of the numeric values of every token. Can be obtained using :class:`~transformers.TapasTokenizer`.
            Only required in case of weak supervision for aggregation (WTQ) to calculate the regression loss.

        Returns:

        Examples::

            >>> from transformers import TapasTokenizer, TapasForQuestionAnswering
            >>> import pandas as pd

            >>> tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-wtq')
            >>> model = TapasForQuestionAnswering.from_pretrained('google/tapas-base-finetuned-wtq')

            >>> data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
            ...         'Age': ["56", "45", "59"],
            ...         'Number of movies': ["87", "53", "69"]
            ... }
            >>> table = pd.DataFrame.from_dict(data)
            >>> queries = ["How many movies has George Clooney played in?", "How old is Brad Pitt?"]

            >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> logits = outputs.logits
            >>> logits_aggregation = outputs.logits_aggregation
        """
        ...
    


@add_start_docstrings("""
    Tapas Model with a sequence classification head on top (a linear layer on top of the pooled output), e.g. for table
    entailment tasks, such as TabFact (Chen et al., 2020).
    """, TAPAS_START_DOCSTRING)
class TapasForSequenceClassification(TapasPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(TAPAS_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> Any | SequenceClassifierOutput:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy). Note: this is called
            "classification_class_index" in the original implementation.

        Returns:

        Examples::

            >>> from transformers import TapasTokenizer, TapasForSequenceClassification
            >>> import torch
            >>> import pandas as pd

            >>> tokenizer = TapasTokenizer.from_pretrained('google/tapas-base-finetuned-tabfact')
            >>> model = TapasForSequenceClassification.from_pretrained('google/tapas-base-finetuned-tabfact')

            >>> data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
            ...         'Age': ["56", "45", "59"],
            ...         'Number of movies': ["87", "53", "69"]
            ... }
            >>> table = pd.DataFrame.from_dict(data)
            >>> queries = ["There is only one actor who is 45 years old", "There are 3 actors which played in more than 60 movies"]

            >>> inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")
            >>> labels = torch.tensor([1, 0]) # 1 means entailed, 0 means refuted

            >>> outputs = model(**inputs, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        ...
    


class AverageApproximationFunction(str, enum.Enum):
    RATIO = ...
    FIRST_ORDER = ...
    SECOND_ORDER = ...


class IndexMap:
    """Index grouping entries within a tensor."""
    def __init__(self, indices, num_segments, batch_dims=...) -> None:
        """
        Creates an index

        Args:
            indices (:obj:`torch.LongTensor`, same shape as a `values` Tensor to which the indices refer):
                Tensor containing the indices.
            num_segments (:obj:`torch.LongTensor`):
                Scalar tensor, the number of segments. All elements in a batched segmented tensor must have the same
                number of segments (although many segments can be empty).
            batch_dims (:obj:`int`, `optional`, defaults to 0):
                The number of batch dimensions. The first `batch_dims` dimensions of a SegmentedTensor are treated as
                batch dimensions. Segments in different batch elements are always distinct even if they have the same
                index.
        """
        ...
    
    def batch_shape(self): # -> Size:
        ...
    


class ProductIndexMap(IndexMap):
    """The product of two indices."""
    def __init__(self, outer_index, inner_index) -> None:
        """
        Combines indices i and j into pairs (i, j). The result is an index where each segment (i, j) is the
        intersection of segments i and j. For example if the inputs represent table cells indexed by respectively rows
        and columns the output will be a table indexed by (row, column) pairs, i.e. by cell. The implementation
        combines indices {0, .., n - 1} and {0, .., m - 1} into {0, .., nm - 1}. The output has `num_segments` equal to
        `outer_index.num_segments` * `inner_index.num_segments`

        Args:
            outer_index (:obj:`IndexMap`):
                IndexMap.
            inner_index (:obj:`IndexMap`):
                IndexMap, must have the same shape as `outer_index`.
        """
        ...
    
    def project_outer(self, index): # -> IndexMap:
        """Projects an index with the same index set onto the outer components."""
        ...
    
    def project_inner(self, index): # -> IndexMap:
        """Projects an index with the same index set onto the inner components."""
        ...
    


def gather(values, index, name=...): # -> Tensor:
    """
    Gathers from `values` using the index map. For each element in the domain of the index map this operation looks up
    a value for that index in `values`. Two elements from the same segment always get assigned the same value.

    Args:
        values (:obj:`torch.Tensor` of shape (B1, ..., Bn, num_segments, V1, ...)):
            Tensor with segment values.
        index (:obj:`IndexMap` of shape (B1, ..., Bn, I1, ..., Ik)):
            IndexMap.
        name (:obj:`str`, `optional`, defaults to 'segmented_gather'):
            Name for the operation. Currently not used

    Returns:
        :obj:`tuple(torch.Tensor)`: Tensor of shape (B1, ..., Bn, I1, ..., Ik, V1, ...) with the gathered values.
    """
    ...

def flatten(index, name=...): # -> IndexMap:
    """
    Flattens a batched index map (which is typically of shape batch_size, seq_length) to a 1d index map. This operation
    relabels the segments to keep batch elements distinct. The k-th batch element will have indices shifted by
    `num_segments` * (k - 1). The result is a tensor with `num_segments` multiplied by the number of elements in the
    batch.

    Args:
        index (:obj:`IndexMap`):
            IndexMap to flatten.
        name (:obj:`str`, `optional`, defaults to 'segmented_flatten'):
            Name for the operation. Currently not used

    Returns:
        (:obj:`IndexMap`): The flattened IndexMap.
    """
    ...

def range_index_map(batch_shape, num_segments, name=...): # -> IndexMap:
    """
    Constructs an index map equal to range(num_segments).

    Args:
        batch_shape (:obj:`torch.Size`):
            Batch shape
        num_segments (:obj:`int`):
            Number of segments
        name (:obj:`str`, `optional`, defaults to 'range_index_map'):
            Name for the operation. Currently not used

    Returns:
        (:obj:`IndexMap`): IndexMap of shape batch_shape with elements equal to range(num_segments).
    """
    ...

def reduce_sum(values, index, name=...): # -> tuple[Unknown, IndexMap]:
    """
    Sums a tensor over its segments.

    Outputs 0 for empty segments.

    This operations computes the sum over segments, with support for:

        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be a sum of
          vectors rather than scalars. Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the sum must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used

    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (:obj:`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments]. .
    """
    ...

def reduce_mean(values, index, name=...): # -> tuple[Unknown, IndexMap]:
    """
    Averages a tensor over its segments.

    Outputs 0 for empty segments.

    This operations computes the mean over segments, with support for:

        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be a mean of
          vectors rather than scalars.

    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the mean must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used

    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (:obj:`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    ...

def reduce_max(values, index, name=...): # -> tuple[Unknown, IndexMap]:
    """
    Computes the maximum over segments.

    This operation computes the maximum over segments, with support for:

        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be an element-wise
          maximum of vectors rather than scalars.

    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the max must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used

    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (:obj:`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    ...

def reduce_min(values, index, name=...): # -> tuple[Unknown, IndexMap]:
    """
    Computes the minimum over segments.

    This operations computes the minimum over segments, with support for:

        - Batching using the first dimensions [B1, B2, ..., Bn]. Each element in a batch can have different indices.
        - Vectorization using the last dimension [V1, V2, ...]. If they are present, the output will be an element-wise
          minimum of vectors rather than scalars.

    Only the middle dimensions [I1, ..., Ik] are reduced by the operation.

    Args:
        values (:obj:`torch.Tensor` of shape [B1, B2, ..., Bn, I1, .., Ik, V1, V2, ..]):
            Tensor containing the values of which the min must be taken segment-wise.
        index (:obj:`IndexMap`, indices are of shape [B1, B2, ..., Bn, I1, .., Ik].):
            Index defining the segments.
        name (:obj:`str`, `optional`, defaults to 'segmented_reduce_sum'):
            Name for the operation. Currently not used

    Returns:
        output_values (:obj:`torch.Tensor`of shape [B1, B2, ..., Bn, num_segments, V1, V2, ..]): Tensor containing the
        output values. output_index (:obj:`IndexMap`): IndexMap with shape [B1, B2, ..., Bn, num_segments].
    """
    ...

def compute_column_logits(sequence_output, column_output_weights, column_output_bias, cell_index, cell_mask, allow_empty_column_selection):
    """
    Computes the column logits.

    Args:
        sequence_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the model.
        column_output_weights (:obj:`torch.FloatTensor` of shape :obj:`(hidden_size)`):
            Weights of the linear layer for column selection.
        column_output_bias (:obj:`torch.FloatTensor` of shape :obj:`()`):
            Bias of the linear layer for column selection.
        cell_index (:obj:`ProductIndexMap`):
            Index that groups tokens into cells.
        cell_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, max_num_rows * max_num_cols)`):
            Mask for cells that exist in the table (i.e. that are not padding).
        allow_empty_column_selection (:obj:`bool`):
            Whether to allow not to select any column

    Returns:
        column_logits (:obj:`torch.FloatTensor`of shape :obj:`(batch_size, max_num_cols)`): Tensor containing the
        column logits for every example in the batch.
    """
    ...

def compute_token_logits(sequence_output, temperature, output_weights, output_bias):
    """
    Computes logits per token

    Args:
        sequence_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the model.
        temperature (:obj:`float`):
            Temperature for the Bernoulli distribution.
        output_weights (:obj:`torch.FloatTensor` of shape :obj:`(hidden_size,)`):
            Weights of the linear layer for cell selection.
        output_bias (:obj:`torch.FloatTensor` of shape :obj:`()`):
            Bias of the linear layer for cell selection

    Returns:
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`): Logits per token.
    """
    ...

def huber_loss(input, target, delta: float = ...): # -> Tensor:
    ...

