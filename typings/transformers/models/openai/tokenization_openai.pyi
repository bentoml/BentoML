

from typing import Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    ...

def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus also does some whitespace standardization
    """
    ...

class OpenAIGPTTokenizer(PreTrainedTokenizer):
    """
    Construct a GPT Tokenizer. Based on Byte-Pair-Encoding with the following peculiarities:

    - lowercases all inputs,
    - uses :obj:`SpaCy` tokenizer and :obj:`ftfy` for pre-BPE tokenization if they are installed, fallback to BERT's
      :obj:`BasicTokenizer` if not.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

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
    def __init__(self, vocab_file, merges_file, unk_token=..., **kwargs) -> None:
        ...
    
    @property
    def do_lower_case(self):
        ...
    
    @property
    def vocab_size(self):
        ...
    
    def get_vocab(self):
        ...
    
    def bpe(self, token):
        ...
    
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


