

from typing import TYPE_CHECKING, List, Optional, Union

from PIL import Image

from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..file_utils import add_end_docstrings, is_torch_available, is_vision_available
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from .base import PIPELINE_INIT_ARGS, Pipeline

if TYPE_CHECKING:
    ...
if is_vision_available():
    ...
if is_torch_available():
    ...
logger = ...
@add_end_docstrings(PIPELINE_INIT_ARGS)
class ImageClassificationPipeline(Pipeline):
    """
    Image classification pipeline using any :obj:`AutoModelForImageClassification`. This pipeline predicts the class of
    an image.

    This image classification pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"image-classification"`.

    See the list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=image-classification>`__.
    """
    def __init__(self, model: Union[PreTrainedModel, TFPreTrainedModel], feature_extractor: PreTrainedFeatureExtractor, framework: Optional[str] = ..., **kwargs) -> None:
        ...
    
    @staticmethod
    def load_image(image: Union[str, Image.Image]):
        ...
    
    def __call__(self, images: Union[str, List[str], Image, List[Image]], top_k=...):
        """
        Assign labels to the image(s) passed as inputs.

        Args:
            images (:obj:`str`, :obj:`List[str]`, :obj:`PIL.Image` or :obj:`List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            top_k (:obj:`int`, `optional`, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            the images.

            The dictionaries contain the following keys:

            - **label** (:obj:`str`) -- The label identified by the model.
            - **score** (:obj:`int`) -- The score attributed by the model for that label.
        """
        ...
    


