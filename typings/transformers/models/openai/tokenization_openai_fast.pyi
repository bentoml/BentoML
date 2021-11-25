

from typing import Optional, Tuple

from ...tokenization_utils_fast import PreTrainedTokenizerFast

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
class OpenAIGPTTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" GPT Tokenizer (backed by HuggingFace's `tokenizers` library). Based on Byte-Pair-Encoding with
    the following peculiarities:

    - lower case all inputs
    - uses BERT's BasicTokenizer for pre-BPE tokenization

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    model_input_names = ...
    slow_tokenizer_class = ...
    def __init__(self, vocab_file=..., merges_file=..., tokenizer_file=..., unk_token=..., **kwargs) -> None:
        ...
    
    @property
    def do_lower_case(self):
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


