

from typing import List, Optional, Union

from ...file_utils import TensorType, add_end_docstrings, add_start_docstrings
from ...tokenization_utils_base import BatchEncoding
from ..bert.tokenization_bert_fast import BertTokenizerFast
from .tokenization_dpr import (
    DPRContextEncoderTokenizer,
    DPRQuestionEncoderTokenizer,
    DPRReaderTokenizer,
)

logger = ...
VOCAB_FILES_NAMES = ...
CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP = ...
QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP = ...
READER_PRETRAINED_VOCAB_FILES_MAP = ...
CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
READER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION = ...
QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION = ...
READER_PRETRAINED_INIT_CONFIGURATION = ...
class DPRContextEncoderTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" DPRContextEncoder tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.DPRContextEncoderTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and
    runs end-to-end tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    pretrained_init_configuration = ...
    slow_tokenizer_class = DPRContextEncoderTokenizer


class DPRQuestionEncoderTokenizerFast(BertTokenizerFast):
    r"""
    Constructs a "fast" DPRQuestionEncoder tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.DPRQuestionEncoderTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and
    runs end-to-end tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    pretrained_init_configuration = ...
    slow_tokenizer_class = DPRQuestionEncoderTokenizer


DPRSpanPrediction = ...
DPRReaderOutput = ...
CUSTOM_DPR_READER_DOCSTRING = ...
@add_start_docstrings(CUSTOM_DPR_READER_DOCSTRING)
class CustomDPRReaderTokenizerMixin:
    def __call__(self, questions, titles: Optional[str] = ..., texts: Optional[str] = ..., padding: Union[bool, str] = ..., truncation: Union[bool, str] = ..., max_length: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., return_attention_mask: Optional[bool] = ..., **kwargs) -> BatchEncoding:
        ...
    
    def decode_best_spans(self, reader_input: BatchEncoding, reader_output: DPRReaderOutput, num_spans: int = ..., max_answer_length: int = ..., num_spans_per_passage: int = ...) -> List[DPRSpanPrediction]:
        """
        Get the span predictions for the extractive Q&A model.

        Returns: `List` of `DPRReaderOutput` sorted by descending `(relevance_score, span_score)`. Each
        `DPRReaderOutput` is a `Tuple` with:

            - **span_score**: ``float`` that corresponds to the score given by the reader for this span compared to
              other spans in the same passage. It corresponds to the sum of the start and end logits of the span.
            - **relevance_score**: ``float`` that corresponds to the score of the each passage to answer the question,
              compared to all the other passages. It corresponds to the output of the QA classifier of the DPRReader.
            - **doc_id**: ``int``` the id of the passage.
            - ***start_index**: ``int`` the start index of the span (inclusive).
            - **end_index**: ``int`` the end index of the span (inclusive).

        Examples::

            >>> from transformers import DPRReader, DPRReaderTokenizer
            >>> tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')
            >>> model = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base')
            >>> encoded_inputs = tokenizer(
            ...         questions=["What is love ?"],
            ...         titles=["Haddaway"],
            ...         texts=["'What Is Love' is a song recorded by the artist Haddaway"],
            ...         return_tensors='pt'
            ...     )
            >>> outputs = model(**encoded_inputs)
            >>> predicted_spans = tokenizer.decode_best_spans(encoded_inputs, outputs)
            >>> print(predicted_spans[0].text)  # best span

        """
        ...
    


@add_end_docstrings(CUSTOM_DPR_READER_DOCSTRING)
class DPRReaderTokenizerFast(CustomDPRReaderTokenizerMixin, BertTokenizerFast):
    r"""
    Constructs a "fast" DPRReader tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.DPRReaderTokenizerFast` is almost identical to :class:`~transformers.BertTokenizerFast` and
    runs end-to-end tokenization: punctuation splitting and wordpiece. The difference is that is has three inputs
    strings: question, titles and texts that are combined to be fed to the :class:`~transformers.DPRReader` model.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.

    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    pretrained_init_configuration = ...
    model_input_names = ...
    slow_tokenizer_class = DPRReaderTokenizer


