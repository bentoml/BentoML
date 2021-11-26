from ..file_utils import add_end_docstrings, is_tf_available, is_torch_available
from .base import PIPELINE_INIT_ARGS, Pipeline

if is_tf_available(): ...
if is_torch_available(): ...

@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        return_all_scores (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to return all prediction scores or just the one of the predicted class.
    """,
)
class TextClassificationPipeline(Pipeline):
    def __init__(self, return_all_scores: bool = ..., **kwargs) -> None: ...
