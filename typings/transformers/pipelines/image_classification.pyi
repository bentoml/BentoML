from typing import TYPE_CHECKING, List, Optional, Union
from PIL import Image
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..file_utils import add_end_docstrings, is_torch_available, is_vision_available
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from .base import PIPELINE_INIT_ARGS, Pipeline

if TYPE_CHECKING: ...
if is_vision_available(): ...
if is_torch_available(): ...
logger = ...

@add_end_docstrings(PIPELINE_INIT_ARGS)
class ImageClassificationPipeline(Pipeline):
    def __init__(
        self,
        model: Union[PreTrainedModel, TFPreTrainedModel],
        feature_extractor: PreTrainedFeatureExtractor,
        framework: Optional[str] = ...,
        **kwargs
    ) -> None: ...
    @staticmethod
    def load_image(image: Union[str, Image.Image]): ...
    def __call__(
        self, images: Union[str, List[str], Image, List[Image]], top_k=...
    ): ...
