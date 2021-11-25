

from typing import Dict, List, Tuple

from tokenizers import Tokenizer

class SentencePieceExtractor:
    """
    Extractor implementation for SentencePiece trained models. https://github.com/google/sentencepiece
    """
    def __init__(self, model: str) -> None:
        ...
    
    def extract(self) -> Tuple[Dict[str, int], List[Tuple]]:
        ...
    


def check_number_comma(piece: str) -> bool:
    ...

class Converter:
    def __init__(self, original_tokenizer) -> None:
        ...
    
    def converted(self) -> Tokenizer:
        ...
    


class BertConverter(Converter):
    def converted(self) -> Tokenizer:
        ...
    


class FunnelConverter(Converter):
    def converted(self) -> Tokenizer:
        ...
    


class MPNetConverter(Converter):
    def converted(self) -> Tokenizer:
        ...
    


class OpenAIGPTConverter(Converter):
    def converted(self) -> Tokenizer:
        ...
    


class GPT2Converter(Converter):
    def converted(self) -> Tokenizer:
        ...
    


class HerbertConverter(Converter):
    def converted(self) -> Tokenizer:
        ...
    


class RobertaConverter(Converter):
    def converted(self) -> Tokenizer:
        ...
    


class RoFormerConverter(Converter):
    def converted(self) -> Tokenizer:
        ...
    


class DebertaConverter(Converter):
    def converted(self) -> Tokenizer:
        ...
    


class SpmConverter(Converter):
    def __init__(self, *args) -> None:
        ...
    
    def vocab(self, proto):
        ...
    
    def unk_id(self, proto):
        ...
    
    def tokenizer(self, proto):
        ...
    
    def normalizer(self, proto):
        ...
    
    def pre_tokenizer(self, replacement, add_prefix_space):
        ...
    
    def post_processor(self):
        ...
    
    def converted(self) -> Tokenizer:
        ...
    


class AlbertConverter(SpmConverter):
    def vocab(self, proto):
        ...
    
    def normalizer(self, proto):
        ...
    
    def post_processor(self):
        ...
    


class BarthezConverter(SpmConverter):
    def unk_id(self, proto):
        ...
    
    def post_processor(self):
        ...
    


class CamembertConverter(SpmConverter):
    def vocab(self, proto):
        ...
    
    def unk_id(self, proto):
        ...
    
    def post_processor(self):
        ...
    


class MBartConverter(SpmConverter):
    def vocab(self, proto):
        ...
    
    def unk_id(self, proto):
        ...
    
    def post_processor(self):
        ...
    


class MBart50Converter(SpmConverter):
    def vocab(self, proto):
        ...
    
    def unk_id(self, proto):
        ...
    
    def post_processor(self):
        ...
    


class XLMRobertaConverter(SpmConverter):
    def vocab(self, proto):
        ...
    
    def unk_id(self, proto):
        ...
    
    def post_processor(self):
        ...
    


class XLNetConverter(SpmConverter):
    def vocab(self, proto):
        ...
    
    def normalizer(self, proto):
        ...
    
    def post_processor(self):
        ...
    


class ReformerConverter(SpmConverter):
    ...


class BertGenerationConverter(SpmConverter):
    ...


class PegasusConverter(SpmConverter):
    def vocab(self, proto):
        ...
    
    def unk_id(self, proto):
        ...
    
    def pre_tokenizer(self, replacement, add_prefix_space):
        ...
    
    def post_processor(self):
        ...
    


class T5Converter(SpmConverter):
    def vocab(self, proto):
        ...
    
    def post_processor(self):
        ...
    


class BigBirdConverter(SpmConverter):
    def post_processor(self):
        ...
    


class CLIPConverter(Converter):
    def converted(self) -> Tokenizer:
        ...
    


SLOW_TO_FAST_CONVERTERS = ...
def convert_slow_tokenizer(transformer_tokenizer) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        transformer_tokenizer (:class:`~transformers.tokenization_utils_base.PreTrainedTokenizer`):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            :class:`~transformers.tokenization_utils_base.PreTrainedTokenizerFast`.

    Return:
        A instance of :class:`~tokenizers.Tokenizer` to be used as the backend tokenizer of a
        :class:`~transformers.tokenization_utils_base.PreTrainedTokenizerFast`
    """
    ...

