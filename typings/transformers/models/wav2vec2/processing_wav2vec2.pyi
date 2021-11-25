

from contextlib import contextmanager

class Wav2Vec2Processor:
    r"""
    Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single
    processor.

    :class:`~transformers.Wav2Vec2Processor` offers all the functionalities of
    :class:`~transformers.Wav2Vec2FeatureExtractor` and :class:`~transformers.Wav2Vec2CTCTokenizer`. See the docstring
    of :meth:`~transformers.Wav2Vec2Processor.__call__` and :meth:`~transformers.Wav2Vec2Processor.decode` for more
    information.

    Args:
        feature_extractor (:obj:`Wav2Vec2FeatureExtractor`):
            An instance of :class:`~transformers.Wav2Vec2FeatureExtractor`. The feature extractor is a required input.
        tokenizer (:obj:`Wav2Vec2CTCTokenizer`):
            An instance of :class:`~transformers.Wav2Vec2CTCTokenizer`. The tokenizer is a required input.
    """
    def __init__(self, feature_extractor, tokenizer) -> None:
        ...
    
    def save_pretrained(self, save_directory):
        """
        Save a Wav2Vec2 feature_extractor object and Wav2Vec2 tokenizer object to the directory ``save_directory``, so
        that it can be re-loaded using the :func:`~transformers.Wav2Vec2Processor.from_pretrained` class method.

        .. note::

            This class method is simply calling
            :meth:`~transformers.feature_extraction_utils.FeatureExtractionMixin.save_pretrained` and
            :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.save_pretrained`. Please refer to the
            docstrings of the methods above for more information.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
        """
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate a :class:`~transformers.Wav2Vec2Processor` from a pretrained Wav2Vec2 processor.

        .. note::

            This class method is simply calling Wav2Vec2FeatureExtractor's
            :meth:`~transformers.feature_extraction_utils.FeatureExtractionMixin.from_pretrained` and
            Wav2Vec2CTCTokenizer's :meth:`~transformers.tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
            Please refer to the docstrings of the methods above for more information.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a feature extractor file saved using the
                  :meth:`~transformers.SequenceFeatureExtractor.save_pretrained` method, e.g.,
                  ``./my_model_directory/``.
                - a path or url to a saved feature extractor JSON `file`, e.g.,
                  ``./my_model_directory/preprocessor_config.json``.
            **kwargs
                Additional keyword arguments passed along to both :class:`~transformers.SequenceFeatureExtractor` and
                :class:`~transformers.PreTrainedTokenizer`
        """
        ...
    
    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        :meth:`~transformers.Wav2Vec2FeatureExtractor.__call__` and returns its output. If used in the context
        :meth:`~transformers.Wav2Vec2Processor.as_target_processor` this method forwards all its arguments to
        Wav2Vec2CTCTokenizer's :meth:`~transformers.Wav2Vec2CTCTokenizer.__call__`. Please refer to the docstring of
        the above two methods for more information.
        """
        ...
    
    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        :meth:`~transformers.Wav2Vec2FeatureExtractor.pad` and returns its output. If used in the context
        :meth:`~transformers.Wav2Vec2Processor.as_target_processor` this method forwards all its arguments to
        Wav2Vec2CTCTokenizer's :meth:`~transformers.Wav2Vec2CTCTokenizer.pad`. Please refer to the docstring of the
        above two methods for more information.
        """
        ...
    
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Wav2Vec2CTCTokenizer's
        :meth:`~transformers.PreTrainedTokenizer.batch_decode`. Please refer to the docstring of this method for more
        information.
        """
        ...
    
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Wav2Vec2CTCTokenizer's
        :meth:`~transformers.PreTrainedTokenizer.decode`. Please refer to the docstring of this method for more
        information.
        """
        ...
    
    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Wav2Vec2.
        """
        ...
    


