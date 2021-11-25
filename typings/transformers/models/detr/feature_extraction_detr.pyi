

import pathlib
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...file_utils import TensorType, is_torch_available
from ...image_utils import ImageFeatureExtractionMixin

"""Feature extractor class for DETR."""
if is_torch_available():
    ...
logger = ...
ImageInput = Union[Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]]
def center_to_corners_format(x): # -> Tensor:
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    """
    ...

def corners_to_center_format(x): # -> ndarray[Unknown, Unknown]:
    """
    Converts a NumPy array of bounding boxes of shape (number of bounding boxes, 4) of corners format (x_0, y_0, x_1,
    y_1) to center format (center_x, center_y, width, height).
    """
    ...

def masks_to_boxes(masks): # -> ndarray[Unknown, Unknown]:
    """
    Compute the bounding boxes around the provided panoptic segmentation masks.

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensor, with the boxes in corner (xyxy) format.
    """
    ...

def rgb_to_id(color): # -> Any | int:
    ...

def id_to_rgb(id_map): # -> ndarray[Unknown, Unknown] | list[Unknown]:
    ...

class DetrFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a DETR feature extractor.

    This feature extractor inherits from :class:`~transformers.FeatureExtractionMixin` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.


    Args:
        format (:obj:`str`, `optional`, defaults to :obj:`"coco_detection"`):
            Data format of the annotations. One of "coco_detection" or "coco_panoptic".
        do_resize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to resize the input to a certain :obj:`size`.
        size (:obj:`int`, `optional`, defaults to 800):
            Resize the input to the given size. Only has an effect if :obj:`do_resize` is set to :obj:`True`. If size
            is a sequence like :obj:`(width, height)`, output size will be matched to this. If size is an int, smaller
            edge of the image will be matched to this number. i.e, if :obj:`height > width`, then image will be
            rescaled to :obj:`(size * height / width, size)`.
        max_size (:obj:`int`, `optional`, defaults to :obj:`1333`):
            The largest size an image dimension can have (otherwise it's capped). Only has an effect if
            :obj:`do_resize` is set to :obj:`True`.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (:obj:`int`, `optional`, defaults to :obj:`[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
        image_std (:obj:`int`, `optional`, defaults to :obj:`[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
            ImageNet std.
    """
    model_input_names = ...
    def __init__(self, format=..., do_resize=..., size=..., max_size=..., do_normalize=..., image_mean=..., image_std=..., **kwargs) -> None:
        ...
    
    def prepare(self, image, target, return_segmentation_masks=..., masks_path=...): # -> tuple[Unknown, Unknown] | tuple[Unknown, dict[Unknown, Unknown]]:
        ...
    
    def convert_coco_poly_to_mask(self, segmentations, height, width): # -> ndarray[Unknown, Unknown]:
        ...
    
    def prepare_coco_detection(self, image, target, return_segmentation_masks=...):
        """
        Convert the target in COCO format into the format expected by DETR.
        """
        ...
    
    def prepare_coco_panoptic(self, image, target, masks_path, return_masks=...): # -> tuple[Unknown, dict[Unknown, Unknown]]:
        ...
    
    def __call__(self, images: ImageInput, annotations: Union[List[Dict], List[List[Dict]]] = ..., return_segmentation_masks: Optional[bool] = ..., masks_path: Optional[pathlib.Path] = ..., pad_and_return_pixel_mask: Optional[bool] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., **kwargs) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s) and optional annotations. Images are by default
        padded up to the largest image in a batch, and a pixel mask is created that indicates which pixels are
        real/which are padding.

        .. warning::

           NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
           PIL images.

        Args:
            images (:obj:`PIL.Image.Image`, :obj:`np.ndarray`, :obj:`torch.Tensor`, :obj:`List[PIL.Image.Image]`, :obj:`List[np.ndarray]`, :obj:`List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            annotations (:obj:`Dict`, :obj:`List[Dict]`, `optional`):
                The corresponding annotations in COCO format.

                In case :class:`~transformers.DetrFeatureExtractor` was initialized with :obj:`format =
                "coco_detection"`, the annotations for each image should have the following format: {'image_id': int,
                'annotations': [annotation]}, with the annotations being a list of COCO object annotations.

                In case :class:`~transformers.DetrFeatureExtractor` was initialized with :obj:`format =
                "coco_panoptic"`, the annotations for each image should have the following format: {'image_id': int,
                'file_name': str, 'segments_info': [segment_info]} with segments_info being a list of COCO panoptic
                annotations.

            return_segmentation_masks (:obj:`Dict`, :obj:`List[Dict]`, `optional`, defaults to :obj:`False`):
                Whether to also include instance segmentation masks as part of the labels in case :obj:`format =
                "coco_detection"`.

            masks_path (:obj:`pathlib.Path`, `optional`):
                Path to the directory containing the PNG files that store the class-agnostic image segmentations. Only
                relevant in case :class:`~transformers.DetrFeatureExtractor` was initialized with :obj:`format =
                "coco_panoptic"`.

            pad_and_return_pixel_mask (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to pad images up to the largest image in a batch and create a pixel mask.

                If left to the default, will return a pixel mask that is:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of NumPy arrays. If set to :obj:`'pt'`, return PyTorch
                :obj:`torch.Tensor` objects.

        Returns:
            :class:`~transformers.BatchFeature`: A :class:`~transformers.BatchFeature` with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when :obj:`pad_and_return_pixel_mask=True` or if
              `"pixel_mask"` is in :obj:`self.model_input_names`).
            - **labels** -- Optional labels to be fed to a model (when :obj:`annotations` are provided)
        """
        ...
    
    def pad_and_create_pixel_mask(self, pixel_values_list: List[torch.Tensor], return_tensors: Optional[Union[str, TensorType]] = ...): # -> BatchFeature:
        """
        Pad images up to the largest image in a batch and create a corresponding :obj:`pixel_mask`.

        Args:
            pixel_values_list (:obj:`List[torch.Tensor]`):
                List of images (pixel values) to be padded. Each image should be a tensor of shape (C, H, W).
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors instead of NumPy arrays. If set to :obj:`'pt'`, return PyTorch
                :obj:`torch.Tensor` objects.

        Returns:
            :class:`~transformers.BatchFeature`: A :class:`~transformers.BatchFeature` with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when :obj:`pad_and_return_pixel_mask=True` or if
              `"pixel_mask"` is in :obj:`self.model_input_names`).

        """
        ...
    
    def post_process(self, outputs, target_sizes): # -> list[dict[str, Unknown]]:
        """
        Converts the output of :class:`~transformers.DetrForObjectDetection` into the format expected by the COCO api.
        Only supports PyTorch.

        Args:
            outputs (:class:`~transformers.DetrObjectDetectionOutput`):
                Raw outputs of the model.
            target_sizes (:obj:`torch.Tensor` of shape :obj:`(batch_size, 2)`, `optional`):
                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation). For visualization, this should be the image size after data
                augment, but before padding.

        Returns:
            :obj:`List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an
            image in the batch as predicted by the model.
        """
        ...
    
    def post_process_segmentation(self, results, outputs, orig_target_sizes, max_target_sizes, threshold=...):
        """
        Converts the output of :class:`~transformers.DetrForSegmentation` into actual instance segmentation
        predictions. Only supports PyTorch.

        Args:
            results (:obj:`List[Dict]`):
                Results list obtained by :meth:`~transformers.DetrFeatureExtractor.post_process`, to which "masks"
                results will be added.
            outputs (:class:`~transformers.DetrSegmentationOutput`):
                Raw outputs of the model.
            orig_target_sizes (:obj:`torch.Tensor` of shape :obj:`(batch_size, 2)`):
                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation).
            max_target_sizes (:obj:`torch.Tensor` of shape :obj:`(batch_size, 2)`):
                Tensor containing the maximum size (h, w) of each image of the batch. For evaluation, this must be the
                original image size (before any data augmentation).
            threshold (:obj:`float`, `optional`, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.

        Returns:
            :obj:`List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels, boxes and masks
            for an image in the batch as predicted by the model.
        """
        ...
    
    def post_process_panoptic(self, outputs, processed_sizes, target_sizes=..., is_thing_map=..., threshold=...):
        """
        Converts the output of :class:`~transformers.DetrForSegmentation` into actual panoptic predictions. Only
        supports PyTorch.

        Parameters:
            outputs (:class:`~transformers.DetrSegmentationOutput`):
                Raw outputs of the model.
            processed_sizes (:obj:`torch.Tensor` of shape :obj:`(batch_size, 2)` or :obj:`List[Tuple]` of length :obj:`batch_size`):
                Torch Tensor (or list) containing the size (h, w) of each image of the batch, i.e. the size after data
                augmentation but before batching.
            target_sizes (:obj:`torch.Tensor` of shape :obj:`(batch_size, 2)` or :obj:`List[Tuple]` of length :obj:`batch_size`, `optional`):
                Torch Tensor (or list) corresponding to the requested final size (h, w) of each prediction. If left to
                None, it will default to the :obj:`processed_sizes`.
            is_thing_map (:obj:`torch.Tensor` of shape :obj:`(batch_size, 2)`, `optional`):
                Dictionary mapping class indices to either True or False, depending on whether or not they are a thing.
                If not set, defaults to the :obj:`is_thing_map` of COCO panoptic.
            threshold (:obj:`float`, `optional`, defaults to 0.85):
                Threshold to use to filter out queries.

        Returns:
            :obj:`List[Dict]`: A list of dictionaries, each dictionary containing a PNG string and segments_info values
            for an image in the batch as predicted by the model.
        """
        ...
    


