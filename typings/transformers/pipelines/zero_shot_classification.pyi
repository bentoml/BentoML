

from typing import List, Union

from ..file_utils import add_end_docstrings
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Pipeline

logger = ...
class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for zero-shot for text classification by turning each possible label into an NLI
    premise/hypothesis pair.
    """
    def __call__(self, sequences, labels, hypothesis_template):
        ...
    


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotClassificationPipeline(Pipeline):
    """
    NLI-based zero-shot classification pipeline using a :obj:`ModelForSequenceClassification` trained on NLI (natural
    language inference) tasks.

    Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis
    pair and passed to the pretrained model. Then, the logit for `entailment` is taken as the logit for the candidate
    label being valid. Any NLI model can be used, but the id of the `entailment` label must be included in the model
    config's :attr:`~transformers.PretrainedConfig.label2id`.

    This NLI pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task identifier:
    :obj:`"zero-shot-classification"`.

    The models that this pipeline can use are models that have been fine-tuned on an NLI task. See the up-to-date list
    of available models on `huggingface.co/models <https://huggingface.co/models?search=nli>`__.
    """
    def __init__(self, args_parser=..., *args, **kwargs) -> None:
        ...
    
    @property
    def entailment_id(self):
        ...
    
    def __call__(self, sequences: Union[str, List[str]], candidate_labels, hypothesis_template=..., multi_label=..., **kwargs):
        """
        Classify the sequence(s) given as inputs. See the :obj:`~transformers.ZeroShotClassificationPipeline`
        documentation for more information.

        Args:
            sequences (:obj:`str` or :obj:`List[str]`):
                The sequence(s) to classify, will be truncated if the model input is too large.
            candidate_labels (:obj:`str` or :obj:`List[str]`):
                The set of possible class labels to classify each sequence into. Can be a single label, a string of
                comma-separated labels, or a list of labels.
            hypothesis_template (:obj:`str`, `optional`, defaults to :obj:`"This example is {}."`):
                The template used to turn each label into an NLI-style hypothesis. This template must include a {} or
                similar syntax for the candidate label to be inserted into the template. For example, the default
                template is :obj:`"This example is {}."` With the candidate label :obj:`"sports"`, this would be fed
                into the model like :obj:`"<cls> sequence to classify <sep> This example is sports . <sep>"`. The
                default template works well in many cases, but it may be worthwhile to experiment with different
                templates depending on the task setting.
            multi_label (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not multiple candidate labels can be true. If :obj:`False`, the scores are normalized such
                that the sum of the label likelihoods for each sequence is 1. If :obj:`True`, the labels are considered
                independent and probabilities are normalized for each candidate by doing a softmax of the entailment
                score vs. the contradiction score.

        Return:
            A :obj:`dict` or a list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **sequence** (:obj:`str`) -- The sequence for which this is the output.
            - **labels** (:obj:`List[str]`) -- The labels sorted by order of likelihood.
            - **scores** (:obj:`List[float]`) -- The probabilities for each of the labels.
        """
        ...
    


