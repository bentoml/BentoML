

from .utils import DataProcessor

logger = ...
class XnliProcessor(DataProcessor):
    """
    Processor for the XNLI dataset. Adapted from
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207
    """
    def __init__(self, language, train_language=...) -> None:
        ...
    
    def get_train_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_test_examples(self, data_dir):
        """See base class."""
        ...
    
    def get_labels(self):
        """See base class."""
        ...
    


xnli_processors = ...
xnli_output_modes = ...
xnli_tasks_num_labels = ...
