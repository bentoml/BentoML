

from enum import Enum
from typing import List, Optional, Union

import tensorflow as tf

from ...file_utils import is_tf_available
from ...tokenization_utils import PreTrainedTokenizer
from .utils import DataProcessor, InputExample

if is_tf_available():
    ...
logger = ...
DEPRECATION_WARNING = ...
def glue_convert_examples_to_features(examples: Union[List[InputExample], tf.data.Dataset], tokenizer: PreTrainedTokenizer, max_length: Optional[int] = ..., task=..., label_list=..., output_mode=...):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset`` containing the
        task-specific features. If the input is a list of ``InputExamples``, will return a list of task-specific
        ``InputFeatures`` which can be fed to the model.

    """
    ...

if is_tf_available():
    ...
class OutputMode(Enum):
    classification = ...
    regression = ...


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        ...
    
    def get_train_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_test_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_labels(self):
        """See base class."""
        ...
    


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        ...
    
    def get_train_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_test_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_labels(self):
        """See base class."""
        ...
    


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_test_examples(self, data_dir):
        """See base class."""
        ...
    


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        ...
    
    def get_train_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_test_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_labels(self):
        """See base class."""
        ...
    


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        ...
    
    def get_train_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_test_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_labels(self):
        """See base class."""
        ...
    


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        ...
    
    def get_train_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_test_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_labels(self):
        """See base class."""
        ...
    


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        ...
    
    def get_train_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_test_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_labels(self):
        """See base class."""
        ...
    


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        ...
    
    def get_train_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_test_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_labels(self):
        """See base class."""
        ...
    


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        ...
    
    def get_train_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_test_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_labels(self):
        """See base class."""
        ...
    


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        ...
    
    def get_train_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_test_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_labels(self):
        """See base class."""
        ...
    


glue_tasks_num_labels = ...
glue_processors = ...
glue_output_modes = ...
