

import os
from typing import Any, Callable, Dict, Set, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey

from .configuration_utils import PretrainedConfig
from .file_utils import PushToHubMixin
from .generation_flax_utils import FlaxGenerationMixin

logger = ...
def quick_gelu(x):
    ...

ACT2FN = ...
class FlaxPreTrainedModel(PushToHubMixin, FlaxGenerationMixin):
    r"""
    Base class for all models.

    :class:`~transformers.FlaxPreTrainedModel` takes care of storing the configuration of the models and handles
    methods for loading, downloading and saving models.

    Class attributes (overridden by derived classes):

        - **config_class** (:class:`~transformers.PretrainedConfig`) -- A subclass of
          :class:`~transformers.PretrainedConfig` to use as configuration class for this model architecture.
        - **base_model_prefix** (:obj:`str`) -- A string indicating the attribute associated to the base model in
          derived classes of the same architecture adding modules on top of the base model.
    """
    config_class = ...
    base_model_prefix = ...
    def __init__(self, config: PretrainedConfig, module: nn.Module, input_shape: Tuple = ..., seed: int = ..., dtype: jnp.dtype = ...) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> Dict:
        ...
    
    @property
    def config(self) -> PretrainedConfig:
        ...
    
    @property
    def module(self) -> nn.Module:
        ...
    
    @property
    def params(self) -> Union[Dict, FrozenDict]:
        ...
    
    @property
    def required_params(self) -> Set:
        ...
    
    @params.setter
    def params(self, params: Union[Dict, FrozenDict]):
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], dtype: jnp.dtype = ..., *model_args, **kwargs):
        r"""
        Instantiate a pretrained flax model from a pre-trained model configuration.

        The warning `Weights from XXX not initialized from pretrained model` means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning `Weights from XXX not used in YYY` means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.FlaxPreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `pt index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In this
                      case, ``from_pt`` should be set to :obj:`True`.
            model_args (sequence of positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
            config (:obj:`Union[PretrainedConfig, str, os.PathLike]`, `optional`):
                Can be either:

                    - an instance of a class derived from :class:`~transformers.PretrainedConfig`,
                    - a string or path valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`.

                Configuration for the model to use instead of an automatically loaded configuation. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the `model id` string of a pretrained
                      model).
                    - The model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named `config.json` is found in the directory.
            cache_dir (:obj:`Union[str, os.PathLike]`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_pt (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Load the model weights from a PyTorch checkpoint save file (see docstring of
                ``pretrained_model_name_or_path`` argument).
            ignore_mismatched_sizes (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str], `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            kwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`). Behaves differently depending on whether a ``config`` is provided or
                automatically loaded:

                    - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
                      underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
                      initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of
                      ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
                      with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
                      attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            >>> from transformers import BertConfig, FlaxBertModel
            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = FlaxBertModel.from_pretrained('bert-base-cased')
            >>> # Model was saved using `save_pretrained('./test/saved_model/')` (for example purposes, not runnable).
            >>> model = FlaxBertModel.from_pretrained('./test/saved_model/')
            >>> # Loading from a PyTorch checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
            >>> config = BertConfig.from_json_file('./pt_model/config.json')
            >>> model = FlaxBertModel.from_pretrained('./pt_model/pytorch_model.bin', from_pt=True, config=config)
        """
        ...
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike[str]], params: Any=..., push_to_hub: bool=..., **kwargs: Any) -> None:
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.FlaxPreTrainedModel.from_pretrained`` class method

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                .. warning::

                    Using :obj:`push_to_hub=True` will synchronize the repository you are pushing to with
                    :obj:`save_directory`, which requires :obj:`save_directory` to be a local clone of the repo you are
                    pushing to if it's an existing folder. Pass along :obj:`temp_dir=True` to use a temporary directory
                    instead.

            kwargs:
                Additional key word arguments passed along to the
                :meth:`~transformers.file_utils.PushToHubMixin.push_to_hub` method.
        """
        ...
    


def overwrite_call_docstring(model_class: object, docstring: str) -> Callable[..., Callable[..., Any]]:
    ...

def append_call_sample_docstring(model_class, tokenizer_class, checkpoint, output_type, config_class, mask=...):
    ...

def append_replace_return_docstrings(model_class, output_type, config_class):
    ...

