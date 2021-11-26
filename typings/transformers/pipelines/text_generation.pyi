from ..file_utils import add_end_docstrings
from .base import PIPELINE_INIT_ARGS, Pipeline

@add_end_docstrings(PIPELINE_INIT_ARGS)
class TextGenerationPipeline(Pipeline):
    XL_PREFIX = ...
    ALLOWED_MODELS = ...
    def __init__(self, *args, return_full_text=..., **kwargs) -> None: ...
