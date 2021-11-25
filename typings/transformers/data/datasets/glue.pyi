

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from torch.utils.data.dataset import Dataset

from ...tokenization_utils_base import PreTrainedTokenizerBase
from ..processors.utils import InputFeatures

logger = ...
@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command
    line.
    """
    task_name: str = ...
    data_dir: str = ...
    max_seq_length: int = ...
    overwrite_cache: bool = ...
    def __post_init__(self): # -> None:
        ...
    


class Split(Enum):
    train = ...
    dev = ...
    test = ...


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]
    def __init__(self, args: GlueDataTrainingArguments, tokenizer: PreTrainedTokenizerBase, limit_length: Optional[int] = ..., mode: Union[str, Split] = ..., cache_dir: Optional[str] = ...) -> None:
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __getitem__(self, i) -> InputFeatures:
        ...
    
    def get_labels(self): # -> list[str] | list[None]:
        ...
    


