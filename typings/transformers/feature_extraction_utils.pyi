

import os
from collections import UserDict
from typing import Any, Dict, Optional, Tuple, Type, Union

from .feature_extraction_sequence_utils import SequenceFeatureExtractor
from .file_utils import TensorType, torch_required

PreTrainedFeatureExtractor: Type[SequenceFeatureExtractor]= ...
class BatchFeature(UserDict[str, Any]):
    r"""
    Holds the output of the :meth:`~transformers.SequenceFeatureExtractor.pad` and feature extractor specific
    ``__call__`` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (:obj:`dict`):
            Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('input_values', 'attention_mask',
            etc.).
        tensor_type (:obj:`Union[None, str, TensorType]`, `optional`):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """
    def __init__(self, data: Optional[Dict[str, Any]] = ..., tensor_type: Union[None, str, TensorType] = ...) -> None:
        ...
    
    def __getitem__(self, item: str) -> Union[Any]:
        """
        If the key is a string, returns the value of the dict associated to :obj:`key` ('input_values',
        'attention_mask', etc.).
        """
        ...
    
    def __getattr__(self, item: str):
        ...
    
    def __getstate__(self):
        ...
    
    def __setstate__(self, state):
        ...
    
    def keys(self):
        ...
    
    def values(self):
        ...
    
    def items(self):
        ...
    
    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = ...):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`):
                The type of tensors to use. If :obj:`str`, should be one of the values of the enum
                :class:`~transformers.file_utils.TensorType`. If :obj:`None`, no modification is done.
        """
        ...
    
    @torch_required
    def to(self, device: Union[str, torch.device]) -> BatchFeature:
        """
        Send all values to device by calling :obj:`v.to(device)` (PyTorch only).

        Args:
            device (:obj:`str` or :obj:`torch.device`): The device to put the tensors on.

        Returns:
            :class:`~transformers.BatchFeature`: The same instance after modification.
        """
        ...
    


class FeatureExtractionMixin:
    """
    This is a feature extraction mixin used to provide saving/loading functionality for sequential and image feature
    extractors.
    """
    def __init__(self, **kwargs) -> None:
        """Set elements of `kwargs` as attributes."""
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> PreTrainedFeatureExtractor:
        r"""
        Instantiate a type of :class:`~transformers.feature_extraction_utils.FeatureExtractionMixin` from a feature
        extractor, *e.g.* a derived class of :class:`~transformers.SequenceFeatureExtractor`.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a feature extractor file saved using the
                  :func:`~transformers.feature_extraction_utils.FeatureExtractionMixin.save_pretrained` method, e.g.,
                  ``./my_model_directory/``.
                - a path or url to a saved feature extractor JSON `file`, e.g.,
                  ``./my_model_directory/preprocessor_config.json``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the feature extractor files and override the cached versions
                if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final feature extractor object. If :obj:`True`,
                then this functions returns a :obj:`Tuple(feature_extractor, unused_kwargs)` where `unused_kwargs` is a
                dictionary consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the
                part of ``kwargs`` which has not been used to update ``feature_extractor`` and is otherwise ignored.
            kwargs (:obj:`Dict[str, Any]`, `optional`):
                The values in kwargs of any keys which are feature extractor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
                controlled by the ``return_unused_kwargs`` keyword parameter.

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.


        Returns:
            A feature extractor of type :class:`~transformers.feature_extraction_utils.FeatureExtractionMixin`.

        Examples::

            # We can't instantiate directly the base class `FeatureExtractionMixin` nor `SequenceFeatureExtractor` so let's show the examples on a
            # derived class: `Wav2Vec2FeatureExtractor`
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h')    # Download feature_extraction_config from huggingface.co and cache.
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('./test/saved_model/')  # E.g. feature_extractor (or model) was saved using `save_pretrained('./test/saved_model/')`
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('./test/saved_model/preprocessor_config.json')
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h', return_attention_mask=False, foo=False)
            assert feature_extractor.return_attention_mask is False
            feature_extractor, unused_kwargs = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base-960h', return_attention_mask=False,
                                                               foo=False, return_unused_kwargs=True)
            assert feature_extractor.return_attention_mask is False
            assert unused_kwargs == {'foo': False}
        """
        ...
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save a feature_extractor object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~transformers.feature_extraction_utils.FeatureExtractionMixin.from_pretrained` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the feature extractor JSON file will be saved (will be created if it does not exist).
        """
        ...
    
    @classmethod
    def get_feature_extractor_dict(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a ``pretrained_model_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
        feature extractor of type :class:`~transformers.feature_extraction_utils.FeatureExtractionMixin` using
        ``from_dict``.

        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature extractor
            object.
        """
        ...
    
    @classmethod
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs) -> PreTrainedFeatureExtractor:
        """
        Instantiates a type of :class:`~transformers.feature_extraction_utils.FeatureExtractionMixin` from a Python
        dictionary of parameters.

        Args:
            feature_extractor_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~transformers.feature_extraction_utils.FeatureExtractionMixin.to_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            :class:`~transformers.feature_extraction_utils.FeatureExtractionMixin`: The feature extractor object
            instantiated from those parameters.
        """
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this feature extractor instance.
        """
        ...
    
    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> PreTrainedFeatureExtractor:
        """
        Instantiates a feature extractor of type :class:`~transformers.feature_extraction_utils.FeatureExtractionMixin`
        from the path to a JSON file of parameters.

        Args:
            json_file (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            A feature extractor of type :class:`~transformers.feature_extraction_utils.FeatureExtractionMixin`: The
            feature_extractor object instantiated from that JSON file.
        """
        ...
    
    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this feature_extractor instance in JSON
            format.
        """
        ...
    
    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this feature_extractor instance's parameters will be saved.
        """
        ...
    
    def __repr__(self):
        ...
    


