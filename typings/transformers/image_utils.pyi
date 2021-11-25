


IMAGENET_DEFAULT_MEAN = ...
IMAGENET_DEFAULT_STD = ...
def is_torch_tensor(obj):
    ...

class ImageFeatureExtractionMixin:
    """
    Mixin that contain utilities for preparing image features.
    """
    def to_pil_image(self, image, rescale=...):
        """
        Converts :obj:`image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last
        axis if needed.

        Args:
            image (:obj:`PIL.Image.Image` or :obj:`numpy.ndarray` or :obj:`torch.Tensor`):
                The image to convert to the PIL Image format.
            rescale (:obj:`bool`, `optional`):
                Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will
                default to :obj:`True` if the image type is a floating type, :obj:`False` otherwise.
        """
        ...
    
    def to_numpy_array(self, image, rescale=..., channel_first=...):
        """
        Converts :obj:`image` to a numpy array. Optionally rescales it and puts the channel dimension as the first
        dimension.

        Args:
            image (:obj:`PIL.Image.Image` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The image to convert to a NumPy array.
            rescale (:obj:`bool`, `optional`):
                Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Will
                default to :obj:`True` if the image is a PIL Image or an array/tensor of integers, :obj:`False`
                otherwise.
            channel_first (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to permute the dimensions of the image to put the channel dimension first.
        """
        ...
    
    def normalize(self, image, mean, std):
        """
        Normalizes :obj:`image` with :obj:`mean` and :obj:`std`. Note that this will trigger a conversion of
        :obj:`image` to a NumPy array if it's a PIL Image.

        Args:
            image (:obj:`PIL.Image.Image` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The image to normalize.
            mean (:obj:`List[float]` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The mean (per channel) to use for normalization.
            std (:obj:`List[float]` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The standard deviation (per channel) to use for normalization.
        """
        ...
    
    def resize(self, image, size, resample=...):
        """
        Resizes :obj:`image`. Note that this will trigger a conversion of :obj:`image` to a PIL Image.

        Args:
            image (:obj:`PIL.Image.Image` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The image to resize.
            size (:obj:`int` or :obj:`Tuple[int, int]`):
                The size to use for resizing the image.
            resample (:obj:`int`, `optional`, defaults to :obj:`PIL.Image.BILINEAR`):
                The filter to user for resampling.
        """
        ...
    
    def center_crop(self, image, size):
        """
        Crops :obj:`image` to the given size using a center crop. Note that if the image is too small to be cropped to
        the size given, it will be padded (so the returned result has the size asked).

        Args:
            image (:obj:`PIL.Image.Image` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The image to resize.
            size (:obj:`int` or :obj:`Tuple[int, int]`):
                The size to which crop the image.
        """
        ...
    


