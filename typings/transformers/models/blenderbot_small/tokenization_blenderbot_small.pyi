

from typing import Dict, List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    ...

class BlenderbotSmallTokenizer(PreTrainedTokenizer):
    """
    Constructs a Blenderbot-90M tokenizer based on BPE (Byte-Pair-Encoding)

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        merges_file (:obj:`str`):
            Path to the merges file.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"__start__"`):
            The beginning of sentence token.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"__end__"`):
            The end of sentence token.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"__unk__"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"__pad__"`):
            The token used for padding, for example when batching sequences of different lengths.
        **kwargs
            Additional keyword arguments passed along to :class:`~transformers.PreTrainedTokenizer`
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    model_input_names = ...
    def __init__(self, vocab_file, merges_file, bos_token=..., eos_token=..., unk_token=..., pad_token=..., **kwargs) -> None:
        ...
    
    @property
    def vocab_size(self) -> int:
        ...
    
    def get_vocab(self) -> Dict:
        ...
    
    def bpe(self, token: str) -> str:
        ...
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens  in a single string."""
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


