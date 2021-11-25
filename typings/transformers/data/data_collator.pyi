

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ..file_utils import PaddingStrategy
from ..modeling_utils import PreTrainedModel
from ..tokenization_utils_base import PreTrainedTokenizerBase

InputDataClass = ...
DataCollator = ...
def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """
    ...

@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = ...
    max_length: Optional[int] = ...
    pad_to_multiple_of: Optional[int] = ...
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        ...
    


@dataclass
class DataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = ...
    max_length: Optional[int] = ...
    pad_to_multiple_of: Optional[int] = ...
    label_pad_token_id: int = ...
    def __call__(self, features):
        ...
    


def tolist(x: Union[List[Any], torch.Tensor]):
    ...

@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = ...
    padding: Union[bool, str, PaddingStrategy] = ...
    max_length: Optional[int] = ...
    pad_to_multiple_of: Optional[int] = ...
    label_pad_token_id: int = ...
    def __call__(self, features):
        ...
    


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = ...
    mlm_probability: float = ...
    pad_to_multiple_of: Optional[int] = ...
    def __post_init__(self):
        ...
    
    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        ...
    
    def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = ...) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        ...
    


@dataclass
class DataCollatorForWholeWordMask(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling that masks entire words.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    .. note::

        This collator relies on details of the implementation of subword tokenization by
        :class:`~transformers.BertTokenizer`, specifically that subword tokens are prefixed with `##`. For tokenizers
        that do not adhere to this scheme, this collator will produce an output that is roughly equivalent to
        :class:`.DataCollatorForLanguageModeling`.
    """
    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        ...
    
    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        ...
    


@dataclass
class DataCollatorForSOP(DataCollatorForLanguageModeling):
    """
    Data collator used for sentence order prediction task.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for both masked language modeling and sentence order prediction
    """
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        ...
    
    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels/attention_mask for masked language modeling: 80% MASK, 10% random, 10%
        original. N-gram not applied yet.
        """
        ...
    


@dataclass
class DataCollatorForPermutationLanguageModeling:
    """
    Data collator used for permutation language modeling.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for permutation language modeling with procedures specific to XLNet
    """
    tokenizer: PreTrainedTokenizerBase
    plm_probability: float = ...
    max_span_length: int = ...
    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        ...
    
    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The masked tokens to be predicted for a particular sequence are determined by the following algorithm:

            0. Start from the beginning of the sequence by setting ``cur_len = 0`` (number of tokens processed so far).
            1. Sample a ``span_length`` from the interval ``[1, max_span_length]`` (length of span of tokens to be
               masked)
            2. Reserve a context of length ``context_length = span_length / plm_probability`` to surround span to be
               masked
            3. Sample a starting point ``start_index`` from the interval ``[cur_len, cur_len + context_length -
               span_length]`` and mask tokens ``start_index:start_index + span_length``
            4. Set ``cur_len = cur_len + context_length``. If ``cur_len < max_len`` (i.e. there are tokens remaining in
               the sequence to be processed), repeat from Step 1.
        """
        ...
    


