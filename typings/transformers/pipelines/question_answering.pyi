

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from ..data import SquadExample
from ..file_utils import add_end_docstrings, is_tf_available, is_torch_available
from ..modelcard import ModelCard
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from ..tokenization_utils import PreTrainedTokenizer
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Pipeline

if TYPE_CHECKING:
    ...
if is_tf_available():
    ...
if is_torch_available():
    ...
class QuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped to
    internal :class:`~transformers.SquadExample`.

    QuestionAnsweringArgumentHandler manages all the possible to create a :class:`~transformers.SquadExample` from the
    command-line supplied arguments.
    """
    def normalize(self, item):
        ...
    
    def __call__(self, *args, **kwargs):
        ...
    


@add_end_docstrings(PIPELINE_INIT_ARGS)
class QuestionAnsweringPipeline(Pipeline):
    """
    Question Answering pipeline using any :obj:`ModelForQuestionAnswering`. See the `question answering examples
    <../task_summary.html#question-answering>`__ for more information.

    This question answering pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a question answering task. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=question-answering>`__.
    """
    default_input_names = ...
    def __init__(self, model: Union[PreTrainedModel, TFPreTrainedModel], tokenizer: PreTrainedTokenizer, modelcard: Optional[ModelCard] = ..., framework: Optional[str] = ..., device: int = ..., task: str = ..., **kwargs) -> None:
        ...
    
    @staticmethod
    def create_sample(question: Union[str, List[str]], context: Union[str, List[str]]) -> Union[SquadExample, List[SquadExample]]:
        """
        QuestionAnsweringPipeline leverages the :class:`~transformers.SquadExample` internally. This helper method
        encapsulate all the logic for converting question(s) and context(s) to :class:`~transformers.SquadExample`.

        We currently support extractive question answering.

        Arguments:
            question (:obj:`str` or :obj:`List[str]`): The question(s) asked.
            context (:obj:`str` or :obj:`List[str]`): The context(s) in which we will look for the answer.

        Returns:
            One or a list of :class:`~transformers.SquadExample`: The corresponding :class:`~transformers.SquadExample`
            grouping question and context.
        """
        ...
    
    def __call__(self, *args, **kwargs):
        """
        Answer the question(s) given as inputs by using the context(s).

        Args:
            args (:class:`~transformers.SquadExample` or a list of :class:`~transformers.SquadExample`):
                One or several :class:`~transformers.SquadExample` containing the question and context.
            X (:class:`~transformers.SquadExample` or a list of :class:`~transformers.SquadExample`, `optional`):
                One or several :class:`~transformers.SquadExample` containing the question and context (will be treated
                the same way as if passed as the first positional argument).
            data (:class:`~transformers.SquadExample` or a list of :class:`~transformers.SquadExample`, `optional`):
                One or several :class:`~transformers.SquadExample` containing the question and context (will be treated
                the same way as if passed as the first positional argument).
            question (:obj:`str` or :obj:`List[str]`):
                One or several question(s) (must be used in conjunction with the :obj:`context` argument).
            context (:obj:`str` or :obj:`List[str]`):
                One or several context(s) associated with the question(s) (must be used in conjunction with the
                :obj:`question` argument).
            topk (:obj:`int`, `optional`, defaults to 1):
                The number of answers to return (will be chosen by order of likelihood). Note that we return less than
                topk answers if there are not enough options available within the context.
            doc_stride (:obj:`int`, `optional`, defaults to 128):
                If the context is too long to fit with the question for the model, it will be split in several chunks
                with some overlap. This argument controls the size of that overlap.
            max_answer_len (:obj:`int`, `optional`, defaults to 15):
                The maximum length of predicted answers (e.g., only answers with a shorter length are considered).
            max_seq_len (:obj:`int`, `optional`, defaults to 384):
                The maximum length of the total sentence (context + question) after tokenization. The context will be
                split in several chunks (using :obj:`doc_stride`) if needed.
            max_question_len (:obj:`int`, `optional`, defaults to 64):
                The maximum length of the question after tokenization. It will be truncated if needed.
            handle_impossible_answer (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not we accept impossible as an answer.

        Return:
            A :obj:`dict` or a list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **score** (:obj:`float`) -- The probability associated to the answer.
            - **start** (:obj:`int`) -- The character start index of the answer (in the tokenized version of the
              input).
            - **end** (:obj:`int`) -- The character end index of the answer (in the tokenized version of the input).
            - **answer** (:obj:`str`) -- The answer to the question.
        """
        ...
    
    def decode(self, start: np.ndarray, end: np.ndarray, topk: int, max_answer_len: int, undesired_tokens: np.ndarray) -> Tuple:
        """
        Take the output of any :obj:`ModelForQuestionAnswering` and will generate probabilities for each span to be the
        actual answer.

        In addition, it filters out some unwanted/impossible cases like answer len being greater than max_answer_len or
        answer end position being before the starting position. The method supports output the k-best answer through
        the topk argument.

        Args:
            start (:obj:`np.ndarray`): Individual start probabilities for each token.
            end (:obj:`np.ndarray`): Individual end probabilities for each token.
            topk (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
            max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
            undesired_tokens (:obj:`np.ndarray`): Mask determining tokens that can be part of the answer
        """
        ...
    
    def span_to_answer(self, text: str, start: int, end: int) -> Dict[str, Union[str, int]]:
        """
        When decoding from token probabilities, this method maps token indexes to actual word in the initial context.

        Args:
            text (:obj:`str`): The actual context to extract the answer from.
            start (:obj:`int`): The answer starting token index.
            end (:obj:`int`): The answer end token index.

        Returns:
            Dictionary like :obj:`{'answer': str, 'start': int, 'end': int}`
        """
        ...
    


