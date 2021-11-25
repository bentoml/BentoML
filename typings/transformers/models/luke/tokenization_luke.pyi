

from typing import Dict, List, Optional, Tuple, Union

from ... import RobertaTokenizer
from ...file_utils import add_end_docstrings
from ...tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    BatchEncoding,
    EncodedInput,
    PaddingStrategy,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)

logger = ...
EntitySpan = Tuple[int, int]
EntitySpanInput = List[EntitySpan]
Entity = str
EntityInput = List[Entity]
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = ...
class LukeTokenizer(RobertaTokenizer):
    r"""
    Construct a LUKE tokenizer.

    This tokenizer inherits from :class:`~transformers.RobertaTokenizer` which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods. Compared to
    :class:`~transformers.RobertaTokenizer`, :class:`~transformers.LukeTokenizer` also creates entity sequences, namely
    :obj:`entity_ids`, :obj:`entity_attention_mask`, :obj:`entity_token_type_ids`, and :obj:`entity_position_ids` to be
    used by the LUKE model.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
        entity_vocab_file (:obj:`str`):
            Path to the entity vocabulary file.
        task (:obj:`str`, `optional`):
            Task for which you want to prepare sequences. One of :obj:`"entity_classification"`,
            :obj:`"entity_pair_classification"`, or :obj:`"entity_span_classification"`. If you specify this argument,
            the entity sequence is automatically created based on the given entity span(s).
        max_entity_length (:obj:`int`, `optional`, defaults to 32):
            The maximum length of :obj:`entity_ids`.
        max_mention_length (:obj:`int`, `optional`, defaults to 30):
            The maximum number of tokens inside an entity span.
        entity_token_1 (:obj:`str`, `optional`, defaults to :obj:`<ent>`):
            The special token used to represent an entity span in a word token sequence. This token is only used when
            ``task`` is set to :obj:`"entity_classification"` or :obj:`"entity_pair_classification"`.
        entity_token_2 (:obj:`str`, `optional`, defaults to :obj:`<ent2>`):
            The special token used to represent an entity span in a word token sequence. This token is only used when
            ``task`` is set to :obj:`"entity_pair_classification"`.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    def __init__(self, vocab_file, merges_file, entity_vocab_file, task=..., max_entity_length=..., max_mention_length=..., entity_token_1=..., entity_token_2=..., **kwargs) -> None:
        ...
    
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(self, text: Union[TextInput, List[TextInput]], text_pair: Optional[Union[TextInput, List[TextInput]]] = ..., entity_spans: Optional[Union[EntitySpanInput, List[EntitySpanInput]]] = ..., entity_spans_pair: Optional[Union[EntitySpanInput, List[EntitySpanInput]]] = ..., entities: Optional[Union[EntityInput, List[EntityInput]]] = ..., entities_pair: Optional[Union[EntityInput, List[EntityInput]]] = ..., add_special_tokens: bool = ..., padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TruncationStrategy] = ..., max_length: Optional[int] = ..., max_entity_length: Optional[int] = ..., stride: int = ..., is_split_into_words: Optional[bool] = ..., pad_to_multiple_of: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., return_token_type_ids: Optional[bool] = ..., return_attention_mask: Optional[bool] = ..., return_overflowing_tokens: bool = ..., return_special_tokens_mask: bool = ..., return_offsets_mapping: bool = ..., return_length: bool = ..., verbose: bool = ..., **kwargs) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences, depending on the task you want to prepare them for.

        Args:
            text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
                tokenizer does not support tokenization based on pretokenized strings.
            text_pair (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence must be a string. Note that this
                tokenizer does not support tokenization based on pretokenized strings.
            entity_spans (:obj:`List[Tuple[int, int]]`, :obj:`List[List[Tuple[int, int]]]`, `optional`):
                The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
                with two integers denoting character-based start and end positions of entities. If you specify
                :obj:`"entity_classification"` or :obj:`"entity_pair_classification"` as the ``task`` argument in the
                constructor, the length of each sequence must be 1 or 2, respectively. If you specify ``entities``, the
                length of each sequence must be equal to the length of each sequence of ``entities``.
            entity_spans_pair (:obj:`List[Tuple[int, int]]`, :obj:`List[List[Tuple[int, int]]]`, `optional`):
                The sequence or batch of sequences of entity spans to be encoded. Each sequence consists of tuples each
                with two integers denoting character-based start and end positions of entities. If you specify the
                ``task`` argument in the constructor, this argument is ignored. If you specify ``entities_pair``, the
                length of each sequence must be equal to the length of each sequence of ``entities_pair``.
            entities (:obj:`List[str]`, :obj:`List[List[str]]`, `optional`):
                The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
                representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
                Angeles). This argument is ignored if you specify the ``task`` argument in the constructor. The length
                of each sequence must be equal to the length of each sequence of ``entity_spans``. If you specify
                ``entity_spans`` without specifying this argument, the entity sequence or the batch of entity sequences
                is automatically constructed by filling it with the [MASK] entity.
            entities_pair (:obj:`List[str]`, :obj:`List[List[str]]`, `optional`):
                The sequence or batch of sequences of entities to be encoded. Each sequence consists of strings
                representing entities, i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los
                Angeles). This argument is ignored if you specify the ``task`` argument in the constructor. The length
                of each sequence must be equal to the length of each sequence of ``entity_spans_pair``. If you specify
                ``entity_spans_pair`` without specifying this argument, the entity sequence or the batch of entity
                sequences is automatically constructed by filling it with the [MASK] entity.
            max_entity_length (:obj:`int`, `optional`):
                The maximum length of :obj:`entity_ids`.
        """
        ...
    
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(self, text: Union[TextInput], text_pair: Optional[Union[TextInput]] = ..., entity_spans: Optional[EntitySpanInput] = ..., entity_spans_pair: Optional[EntitySpanInput] = ..., entities: Optional[EntityInput] = ..., entities_pair: Optional[EntityInput] = ..., add_special_tokens: bool = ..., padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TruncationStrategy] = ..., max_length: Optional[int] = ..., max_entity_length: Optional[int] = ..., stride: int = ..., is_split_into_words: Optional[bool] = ..., pad_to_multiple_of: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., return_token_type_ids: Optional[bool] = ..., return_attention_mask: Optional[bool] = ..., return_overflowing_tokens: bool = ..., return_special_tokens_mask: bool = ..., return_offsets_mapping: bool = ..., return_length: bool = ..., verbose: bool = ..., **kwargs) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences.

        .. warning:: This method is deprecated, ``__call__`` should be used instead.

        Args:
            text (:obj:`str`):
                The first sequence to be encoded. Each sequence must be a string.
            text_pair (:obj:`str`):
                The second sequence to be encoded. Each sequence must be a string.
            entity_spans (:obj:`List[Tuple[int, int]]`, :obj:`List[List[Tuple[int, int]]]`, `optional`)::
                The first sequence of entity spans to be encoded. The sequence consists of tuples each with two
                integers denoting character-based start and end positions of entities. If you specify
                :obj:`"entity_classification"` or :obj:`"entity_pair_classification"` as the ``task`` argument in the
                constructor, the length of each sequence must be 1 or 2, respectively. If you specify ``entities``, the
                length of the sequence must be equal to the length of ``entities``.
            entity_spans_pair (:obj:`List[Tuple[int, int]]`, :obj:`List[List[Tuple[int, int]]]`, `optional`)::
                The second sequence of entity spans to be encoded. The sequence consists of tuples each with two
                integers denoting character-based start and end positions of entities. If you specify the ``task``
                argument in the constructor, this argument is ignored. If you specify ``entities_pair``, the length of
                the sequence must be equal to the length of ``entities_pair``.
            entities (:obj:`List[str]` `optional`)::
                The first sequence of entities to be encoded. The sequence consists of strings representing entities,
                i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los Angeles). This argument
                is ignored if you specify the ``task`` argument in the constructor. The length of the sequence must be
                equal to the length of ``entity_spans``. If you specify ``entity_spans`` without specifying this
                argument, the entity sequence is automatically constructed by filling it with the [MASK] entity.
            entities_pair (:obj:`List[str]`, :obj:`List[List[str]]`, `optional`)::
                The second sequence of entities to be encoded. The sequence consists of strings representing entities,
                i.e., special entities (e.g., [MASK]) or entity titles of Wikipedia (e.g., Los Angeles). This argument
                is ignored if you specify the ``task`` argument in the constructor. The length of the sequence must be
                equal to the length of ``entity_spans_pair``. If you specify ``entity_spans_pair`` without specifying
                this argument, the entity sequence is automatically constructed by filling it with the [MASK] entity.
            max_entity_length (:obj:`int`, `optional`):
                The maximum length of the entity sequence.
        """
        ...
    
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def batch_encode_plus(self, batch_text_or_text_pairs: Union[List[TextInput], List[TextInputPair]], batch_entity_spans_or_entity_spans_pairs: Optional[Union[List[EntitySpanInput], List[Tuple[EntitySpanInput, EntitySpanInput]]]] = ..., batch_entities_or_entities_pairs: Optional[Union[List[EntityInput], List[Tuple[EntityInput, EntityInput]]]] = ..., add_special_tokens: bool = ..., padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TruncationStrategy] = ..., max_length: Optional[int] = ..., max_entity_length: Optional[int] = ..., stride: int = ..., is_split_into_words: Optional[bool] = ..., pad_to_multiple_of: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., return_token_type_ids: Optional[bool] = ..., return_attention_mask: Optional[bool] = ..., return_overflowing_tokens: bool = ..., return_special_tokens_mask: bool = ..., return_offsets_mapping: bool = ..., return_length: bool = ..., verbose: bool = ..., **kwargs) -> BatchEncoding:
        """
        Tokenize and prepare for the model a list of sequences or a list of pairs of sequences.

        .. warning::
            This method is deprecated, ``__call__`` should be used instead.


        Args:
            batch_text_or_text_pairs (:obj:`List[str]`, :obj:`List[Tuple[str, str]]`):
                Batch of sequences or pair of sequences to be encoded. This can be a list of string or a list of pair
                of string (see details in ``encode_plus``).
            batch_entity_spans_or_entity_spans_pairs (:obj:`List[List[Tuple[int, int]]]`,
            :obj:`List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]`, `optional`)::
                Batch of entity span sequences or pairs of entity span sequences to be encoded (see details in
                ``encode_plus``).
            batch_entities_or_entities_pairs (:obj:`List[List[str]]`, :obj:`List[Tuple[List[str], List[str]]]`,
            `optional`):
                Batch of entity sequences or pairs of entity sequences to be encoded (see details in ``encode_plus``).
            max_entity_length (:obj:`int`, `optional`):
                The maximum length of the entity sequence.
        """
        ...
    
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(self, ids: List[int], pair_ids: Optional[List[int]] = ..., entity_ids: Optional[List[int]] = ..., pair_entity_ids: Optional[List[int]] = ..., entity_token_spans: Optional[List[Tuple[int, int]]] = ..., pair_entity_token_spans: Optional[List[Tuple[int, int]]] = ..., add_special_tokens: bool = ..., padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TruncationStrategy] = ..., max_length: Optional[int] = ..., max_entity_length: Optional[int] = ..., stride: int = ..., pad_to_multiple_of: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., return_token_type_ids: Optional[bool] = ..., return_attention_mask: Optional[bool] = ..., return_overflowing_tokens: bool = ..., return_special_tokens_mask: bool = ..., return_offsets_mapping: bool = ..., return_length: bool = ..., verbose: bool = ..., prepend_batch_axis: bool = ..., **kwargs) -> BatchEncoding:
        """
        Prepares a sequence of input id, entity id and entity span, or a pair of sequences of inputs ids, entity ids,
        entity spans so that it can be used by the model. It adds special tokens, truncates sequences if overflowing
        while taking into account the special tokens and manages a moving window (with user defined stride) for
        overflowing tokens


        Args:
            ids (:obj:`List[int]`):
                Tokenized input ids of the first sequence.
            pair_ids (:obj:`List[int]`, `optional`):
                Tokenized input ids of the second sequence.
            entity_ids (:obj:`List[int]`, `optional`):
                Entity ids of the first sequence.
            pair_entity_ids (:obj:`List[int]`, `optional`):
                Entity ids of the second sequence.
            entity_token_spans (:obj:`List[Tuple[int, int]]`, `optional`):
                Entity spans of the first sequence.
            pair_entity_token_spans (:obj:`List[Tuple[int, int]]`, `optional`):
                Entity spans of the second sequence.
            max_entity_length (:obj:`int`, `optional`):
                The maximum length of the entity sequence.
        """
        ...
    
    def pad(self, encoded_inputs: Union[BatchEncoding, List[BatchEncoding], Dict[str, EncodedInput], Dict[str, List[EncodedInput]], List[Dict[str, EncodedInput]]], , padding: Union[bool, str, PaddingStrategy] = ..., max_length: Optional[int] = ..., max_entity_length: Optional[int] = ..., pad_to_multiple_of: Optional[int] = ..., return_attention_mask: Optional[bool] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., verbose: bool = ...) -> BatchEncoding:
        """
        Pad a single encoded input or a batch of encoded inputs up to predefined length or to the max sequence length
        in the batch. Padding side (left/right) padding token ids are defined at the tokenizer level (with
        ``self.padding_side``, ``self.pad_token_id`` and ``self.pad_token_type_id``) .. note:: If the
        ``encoded_inputs`` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the result
        will use the same type unless you provide a different tensor type with ``return_tensors``. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        Args:
            encoded_inputs (:class:`~transformers.BatchEncoding`, list of :class:`~transformers.BatchEncoding`, :obj:`Dict[str, List[int]]`, :obj:`Dict[str, List[List[int]]` or :obj:`List[Dict[str, List[int]]]`):
                Tokenized inputs. Can represent one input (:class:`~transformers.BatchEncoding` or :obj:`Dict[str,
                List[int]]`) or a batch of tokenized inputs (list of :class:`~transformers.BatchEncoding`, `Dict[str,
                List[List[int]]]` or `List[Dict[str, List[int]]]`) so you can use this method during preprocessing as
                well as in a PyTorch Dataloader collate function. Instead of :obj:`List[int]` you can have tensors
                (numpy arrays, PyTorch tensors or TensorFlow tensors), see the note above for the return type.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                 Select a strategy to pad the returned sequences (according to the model's padding side and padding
                 index) among:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the returned list and optionally padding length (see above).
            max_entity_length (:obj:`int`, `optional`):
                The maximum length of the entity sequence.
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific tokenizer's default, defined by the :obj:`return_outputs` attribute. `What are
                attention masks? <../glossary.html#attention-mask>`__
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to print more information and warnings.
        """
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


