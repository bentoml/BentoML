

from contextlib import contextmanager
from typing import List, Optional

from ...tokenization_utils_base import BatchEncoding

logger = ...
class RagTokenizer:
    def __init__(self, question_encoder, generator) -> None:
        ...
    
    def save_pretrained(self, save_directory):
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        ...
    
    def __call__(self, *args, **kwargs):
        ...
    
    def batch_decode(self, *args, **kwargs):
        ...
    
    def decode(self, *args, **kwargs):
        ...
    
    @contextmanager
    def as_target_tokenizer(self):
        """
        Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
        sequence-to-sequence models that need a slightly different processing for the labels.
        """
        ...
    
    def prepare_seq2seq_batch(self, src_texts: List[str], tgt_texts: Optional[List[str]] = ..., max_length: Optional[int] = ..., max_target_length: Optional[int] = ..., padding: str = ..., return_tensors: str = ..., truncation: bool = ..., **kwargs) -> BatchEncoding:
        ...
    


