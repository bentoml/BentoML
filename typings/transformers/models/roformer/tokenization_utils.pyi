

from typing import List

from tokenizers import NormalizedString, PreTokenizedString

"""Tokenization utils for RoFormer."""
class JiebaPreTokenizer:
    def __init__(self, vocab) -> None:
        ...
    
    def jieba_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        ...
    
    def pre_tokenize(self, pretok: PreTokenizedString): # -> None:
        ...
    


