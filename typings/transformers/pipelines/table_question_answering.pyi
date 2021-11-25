

from ..file_utils import add_end_docstrings, is_torch_available
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Pipeline

if is_torch_available():
    ...
class TableQuestionAnsweringArgumentHandler(ArgumentHandler):
    """
    Handles arguments for the TableQuestionAnsweringPipeline
    """
    def __call__(self, table=..., query=..., sequential=..., padding=..., truncation=...):
        ...
    


@add_end_docstrings(PIPELINE_INIT_ARGS)
class TableQuestionAnsweringPipeline(Pipeline):
    """
    Table Question Answering pipeline using a :obj:`ModelForTableQuestionAnswering`. This pipeline is only available in
    PyTorch.

    This tabular question answering pipeline can currently be loaded from :func:`~transformers.pipeline` using the
    following task identifier: :obj:`"table-question-answering"`.

    The models that this pipeline can use are models that have been fine-tuned on a tabular question answering task.
    See the up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=table-question-answering>`__.
    """
    default_input_names = ...
    def __init__(self, args_parser=..., *args, **kwargs) -> None:
        ...
    
    def batch_inference(self, **inputs):
        ...
    
    def sequential_inference(self, **inputs):
        """
        Inference used for models that need to process sequences in a sequential fashion, like the SQA models which
        handle conversational query related to a table.
        """
        ...
    
    def __call__(self, *args, **kwargs):
        r"""
        Answers queries according to a table. The pipeline accepts several types of inputs which are detailed below:

        - ``pipeline(table, query)``
        - ``pipeline(table, [query])``
        - ``pipeline(table=table, query=query)``
        - ``pipeline(table=table, query=[query])``
        - ``pipeline({"table": table, "query": query})``
        - ``pipeline({"table": table, "query": [query]})``
        - ``pipeline([{"table": table, "query": query}, {"table": table, "query": query}])``

        The :obj:`table` argument should be a dict or a DataFrame built from that dict, containing the whole table:

        Example::

            data = {
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            }

        This dictionary can be passed in as such, or can be converted to a pandas DataFrame:

        Example::

            import pandas as pd
            table = pd.DataFrame.from_dict(data)


        Args:
            table (:obj:`pd.DataFrame` or :obj:`Dict`):
                Pandas DataFrame or dictionary that will be converted to a DataFrame containing all the table values.
                See above for an example of dictionary.
            query (:obj:`str` or :obj:`List[str]`):
                Query or list of queries that will be sent to the model alongside the table.
            sequential (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to do inference sequentially or as a batch. Batching is faster, but models like SQA require the
                inference to be done sequentially to extract relations within sequences, given their conversational
                nature.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls padding. Accepts the following values:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).

            truncation (:obj:`bool`, :obj:`str` or :class:`~transformers.TapasTruncationStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls truncation. Accepts the following values:

                * :obj:`True` or :obj:`'drop_rows_to_fit'`: Truncate to a maximum length specified with the argument
                  :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not
                  provided. This will truncate row by row, removing rows from the table.
                * :obj:`False` or :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with
                  sequence lengths greater than the model maximum admissible input size).


        Return:
            A dictionary or a list of dictionaries containing results: Each result is a dictionary with the following
            keys:

            - **answer** (:obj:`str`) -- The answer of the query given the table. If there is an aggregator, the answer
              will be preceded by :obj:`AGGREGATOR >`.
            - **coordinates** (:obj:`List[Tuple[int, int]]`) -- Coordinates of the cells of the answers.
            - **cells** (:obj:`List[str]`) -- List of strings made up of the answer cell values.
            - **aggregator** (:obj:`str`) -- If the model has an aggregator, this returns the aggregator.
        """
        ...
    


