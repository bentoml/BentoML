

from ..xlm.tokenization_xlm import XLMTokenizer

logger = ...
VOCAB_FILES_NAMES = ...
PRETRAINED_VOCAB_FILES_MAP = ...
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = ...
PRETRAINED_INIT_CONFIGURATION = ...
def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """
    ...

class FlaubertTokenizer(XLMTokenizer):
    """
    Construct a Flaubert tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses preprocessing and tokenization.
    - Normalizing all inputs text.
    - The arguments ``special_tokens`` and the function ``set_special_tokens``, can be used to add additional symbols
      (like "__classify__") to a vocabulary.
    - The argument :obj:`do_lowercase` controls lower casing (automatically set for pretrained vocabularies).

    This tokenizer inherits from :class:`~transformers.XLMTokenizer`. Please check the superclass for usage examples
    and documentation regarding arguments.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    pretrained_init_configuration = ...
    max_model_input_sizes = ...
    def __init__(self, do_lowercase=..., **kwargs) -> None:
        ...
    
    def preprocess_text(self, text):
        ...
    


