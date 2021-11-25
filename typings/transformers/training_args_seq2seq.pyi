

from dataclasses import dataclass

from .file_utils import add_start_docstrings
from .training_args import TrainingArguments

logger = ...
@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use a `sortish sampler` or not. Only possible if the underlying datasets are `Seq2SeqDataset` for
        now but will become generally available in the near future.

        It sorts the inputs according to lengths in order to minimize the padding size, with a bit of randomness for
        the training set.
    predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """
    sortish_sampler: bool = ...
    predict_with_generate: bool = ...


