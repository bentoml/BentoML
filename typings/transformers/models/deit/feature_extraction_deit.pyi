

from typing import List, Optional, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...file_utils import TensorType
from ...image_utils import ImageFeatureExtractionMixin

"""Feature extractor class for DeiT."""
logger = ...
class DeiTFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a DeiT feature extractor.

    This feature extractor inherits from :class:`~transformers.FeatureExtractionMixin` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to resize the input to a certain :obj:`size`.
        size (:obj:`int` or :obj:`Tuple(int)`, `optional`, defaults to 256):
            Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
            integer is provided, then the input will be resized to (size, size). Only has an effect if :obj:`do_resize`
            is set to :obj:`True`.
        resample (:obj:`int`, `optional`, defaults to :obj:`PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of :obj:`PIL.Image.NEAREST`, :obj:`PIL.Image.BOX`,
            :obj:`PIL.Image.BILINEAR`, :obj:`PIL.Image.HAMMING`, :obj:`PIL.Image.BICUBIC` or :obj:`PIL.Image.LANCZOS`.
            Only has an effect if :obj:`do_resize` is set to :obj:`True`.
        do_center_crop (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to crop the input at the center. If the input size is smaller than :obj:`crop_size` along any edge,
            the image is padded with 0's and then center cropped.
        crop_size (:obj:`int`, `optional`, defaults to 224):
            Desired output size when applying center-cropping. Only has an effect if :obj:`do_center_crop` is set to
            :obj:`True`.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to normalize the input with :obj:`image_mean` and :obj:`image_std`.
        image_mean (:obj:`List[int]`, defaults to :obj:`[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (:obj:`List[int]`, defaults to :obj:`[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """
    model_input_names = ...
    def __init__(self, do_resize=..., size=..., resample=..., do_center_crop=..., crop_size=..., do_normalize=..., image_mean=..., image_std=..., **kwargs) -> None:
        ...
    
    def __call__(self, images: Union[Image.Image, np.ndarray, torch.Tensor, List[Image.Image], List[np.ndarray], List[torch.Tensor]], return_tensors: Optional[Union[str, TensorType]] = ..., **kwargs) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        .. warning::

           NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
           PIL images.

        Args:
            images (:obj:`PIL.Image.Image`, :obj:`np.ndarray`, :obj:`torch.Tensor`, :obj:`List[PIL.Image.Image]`, :obj:`List[np.ndarray]`, :obj:`List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`, defaults to :obj:`'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return NumPy :obj:`np.ndarray` objects.
                * :obj:`'jax'`: Return JAX :obj:`jnp.ndarray` objects.

        Returns:
            :class:`~transformers.BatchFeature`: A :class:`~transformers.BatchFeature` with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
              width).
        """
        ...
    


