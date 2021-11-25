

from typing import Dict, List, Optional

import torch
from torch.utils.data.dataset import Dataset

from ...tokenization_utils import PreTrainedTokenizer

logger = ...
DEPRECATION_WARNING = ...
class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=..., cache_dir: Optional[str] = ...) -> None:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __getitem__(self, i) -> torch.Tensor:
        ...
    


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int) -> None:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        ...
    


class LineByLineWithRefDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, ref_path: str) -> None:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        ...
    


class LineByLineWithSOPTextDataset(Dataset):
    """
    Dataset for sentence order prediction task, prepare sentence pairs for SOP task
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_dir: str, block_size: int) -> None:
        ...
    
    def create_examples_from_document(self, document, block_size, tokenizer, short_seq_prob=...):
        """Creates examples for a single document."""
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        ...
    


class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=..., short_seq_probability=..., nsp_probability=...) -> None:
        ...
    
    def create_examples_from_document(self, document: List[List[int]], doc_index: int, block_size: int): # -> None:
        """Creates examples for a single document."""
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __getitem__(self, i): # -> Any:
        ...
    


