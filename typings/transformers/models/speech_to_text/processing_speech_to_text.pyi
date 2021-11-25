

from contextlib import contextmanager

"""
Speech processor class for Speech2Text
"""
class Speech2TextProcessor:
    r"""
    Constructs a Speech2Text processor which wraps a Speech2Text feature extractor and a Speech2Text tokenizer into a
    single processor.

    :class:`~transformers.Speech2TextProcessor` offers all the functionalities of
    :class:`~transformers.Speech2TextFeatureExtractor` and :class:`~transformers.Speech2TextTokenizer`. See the
    :meth:`~transformers.Speech2TextProcessor.__call__` and :meth:`~transformers.Speech2TextProcessor.decode` for more
    information.

    Args:
        feature_extractor (:obj:`Speech2TextFeatureExtractor`):
            An instance of :class:`~transformers.Speech2TextFeatureExtractor`. The feature extractor is a required
            input.
        tokenizer (:obj:`Speech2TextTokenizer`):
            An instance of :class:`~transformers.Speech2TextTokenizer`. The tokenizer is a required input.
    """
    def __init__(self, feature_extractor, tokenizer) -> None:
        ...
    
    def save_pretrained(self, save_directory): # -> None:
        """
        Save a Speech2Text feature extractor object and Speech2Text tokenizer object to the directory
        ``save_directory``, so that it can be re-loaded using the
        :func:`~transformers.Speech2TextProcessor.from_pretrained` class method.

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
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs): # -> Self@Speech2TextProcessor:
        r"""
        Instantiate a :class:`~transformers.Speech2TextProcessor` from a pretrained Speech2Text processor.

        .. note::

            This class method is simply calling Speech2TextFeatureExtractor's
            :meth:`~transformers.PreTrainedFeatureExtractor.from_pretrained` and Speech2TextTokenizer's
            :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.from_pretrained`. Please refer to the
            docstrings of the methods above for more information.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
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
    
    def __call__(self, *args, **kwargs): # -> BatchFeature | BatchEncoding:
        """
        When used in normal mode, this method forwards all its arguments to Speech2TextFeatureExtractor's
        :meth:`~transformers.Speech2TextFeatureExtractor.__call__` and returns its output. If used in the context
        :meth:`~transformers.Speech2TextProcessor.as_target_processor` this method forwards all its arguments to
        Speech2TextTokenizer's :meth:`~transformers.Speech2TextTokenizer.__call__`. Please refer to the doctsring of
        the above two methods for more information.
        """
        ...
    
    def batch_decode(self, *args, **kwargs): # -> List[str]:
        """
        This method forwards all its arguments to Speech2TextTokenizer's
        :meth:`~transformers.PreTrainedTokenizer.batch_decode`. Please refer to the docstring of this method for more
        information.
        """
        ...
    
    def decode(self, *args, **kwargs): # -> str:
        """
        This method forwards all its arguments to Speech2TextTokenizer's
        :meth:`~transformers.PreTrainedTokenizer.decode`. Please refer to the docstring of this method for more
        information.
        """
        ...
    
    @contextmanager
    def as_target_processor(self): # -> Generator[None, None, None]:
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Speech2Text.
        """
        ...
    


