

"""
Image/Text processor class for CLIP
"""
class CLIPProcessor:
    r"""
    Constructs a CLIP processor which wraps a CLIP feature extractor and a CLIP tokenizer into a single processor.

    :class:`~transformers.CLIPProcessor` offers all the functionalities of :class:`~transformers.CLIPFeatureExtractor`
    and :class:`~transformers.CLIPTokenizer`. See the :meth:`~transformers.CLIPProcessor.__call__` and
    :meth:`~transformers.CLIPProcessor.decode` for more information.

    Args:
        feature_extractor (:class:`~transformers.CLIPFeatureExtractor`):
            The feature extractor is a required input.
        tokenizer (:class:`~transformers.CLIPTokenizer`):
            The tokenizer is a required input.
    """
    def __init__(self, feature_extractor, tokenizer) -> None:
        ...
    
    def save_pretrained(self, save_directory): # -> None:
        """
        Save a CLIP feature extractor object and CLIP tokenizer object to the directory ``save_directory``, so that it
        can be re-loaded using the :func:`~transformers.CLIPProcessor.from_pretrained` class method.

        .. note::

            This class method is simply calling :meth:`~transformers.PreTrainedFeatureExtractor.save_pretrained` and
            :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.save_pretrained`. Please refer to the
            docstrings of the methods above for more information.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs): # -> Self@CLIPProcessor:
        r"""
        Instantiate a :class:`~transformers.CLIPProcessor` from a pretrained CLIP processor.

        .. note::

            This class method is simply calling CLIPFeatureExtractor's
            :meth:`~transformers.PreTrainedFeatureExtractor.from_pretrained` and CLIPTokenizer's
            :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.from_pretrained`. Please refer to the
            docstrings of the methods above for more information.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``clip-vit-base-patch32``, or
                  namespaced under a user or organization name, like ``openai/clip-vit-base-patch32``.
                - a path to a `directory` containing a feature extractor file saved using the
                  :meth:`~transformers.PreTrainedFeatureExtractor.save_pretrained` method, e.g.,
                  ``./my_model_directory/``.
                - a path or url to a saved feature extractor JSON `file`, e.g.,
                  ``./my_model_directory/preprocessor_config.json``.

            **kwargs
                Additional keyword arguments passed along to both :class:`~transformers.PreTrainedFeatureExtractor` and
                :class:`~transformers.PreTrainedTokenizer`
        """
        ...
    
    def __call__(self, text=..., images=..., return_tensors=..., **kwargs): # -> BatchEncoding:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the
        :obj:`text` and :obj:`kwargs` arguments to CLIPTokenizer's :meth:`~transformers.CLIPTokenizer.__call__` if
        :obj:`text` is not :obj:`None` to encode the text. To prepare the image(s), this method forwards the
        :obj:`images` and :obj:`kwrags` arguments to CLIPFeatureExtractor's
        :meth:`~transformers.CLIPFeatureExtractor.__call__` if :obj:`images` is not :obj:`None`. Please refer to the
        doctsring of the above two methods for more information.

        Args:
            text (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                :obj:`is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (:obj:`PIL.Image.Image`, :obj:`np.ndarray`, :obj:`torch.Tensor`, :obj:`List[PIL.Image.Image]`, :obj:`List[np.ndarray]`, :obj:`List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                If set, will return tensors of a particular framework. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return NumPy :obj:`np.ndarray` objects.
                * :obj:`'jax'`: Return JAX :obj:`jnp.ndarray` objects.

        Returns:
            :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when :obj:`text` is not :obj:`None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              :obj:`return_attention_mask=True` or if `"attention_mask"` is in :obj:`self.model_input_names` and if
              :obj:`text` is not :obj:`None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when :obj:`images` is not :obj:`None`.
        """
        ...
    
    def batch_decode(self, *args, **kwargs): # -> List[str]:
        """
        This method forwards all its arguments to CLIPTokenizer's
        :meth:`~transformers.PreTrainedTokenizer.batch_decode`. Please refer to the docstring of this method for more
        information.
        """
        ...
    
    def decode(self, *args, **kwargs): # -> str:
        """
        This method forwards all its arguments to CLIPTokenizer's :meth:`~transformers.PreTrainedTokenizer.decode`.
        Please refer to the docstring of this method for more information.
        """
        ...
    


