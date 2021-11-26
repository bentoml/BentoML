from typing import List, Union
from ..file_utils import add_end_docstrings
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Pipeline

logger = ...

class ZeroShotClassificationArgumentHandler(ArgumentHandler):
    def __call__(self, sequences, labels, hypothesis_template): ...

@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotClassificationPipeline(Pipeline):
    def __init__(self, args_parser=..., *args, **kwargs) -> None: ...
    @property
    def entailment_id(self): ...
    def __call__(
        self,
        sequences: Union[str, List[str]],
        candidate_labels,
        hypothesis_template=...,
        multi_label=...,
        **kwargs
    ): ...
