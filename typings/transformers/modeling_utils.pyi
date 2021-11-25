

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor, device, nn

from .configuration_utils import PretrainedConfig
from .file_utils import ModelOutput, PushToHubMixin, replace_return_docstrings
from .generation_utils import GenerationMixin

logger = ...
_init_weights = ...
@contextmanager
def no_init_weights(_enable=...):
    """
    Context manager to globally disable weight initialization to speed up loading large models.

    TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
    """
    ...

def find_pruneable_heads_and_indices(heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.

    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    ...

def get_parameter_device(parameter: Union[nn.Module, GenerationMixin, ModuleUtilsMixin]):
    ...

def get_parameter_dtype(parameter: Union[nn.Module, GenerationMixin, ModuleUtilsMixin]):
    ...

class ModuleUtilsMixin:
    """
    A few utilities for :obj:`torch.nn.Modules`, to be used as a mixin.
    """
    def add_memory_hooks(self):
        """
        Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.

        Increase in memory consumption is stored in a :obj:`mem_rss_diff` attribute for each module and can be reset to
        zero with :obj:`model.reset_memory_hooks_state()`.
        """
        ...
    
    def reset_memory_hooks_state(self):
        """
        Reset the :obj:`mem_rss_diff` attribute of each module (see
        :func:`~transformers.modeling_utils.ModuleUtilsMixin.add_memory_hooks`).
        """
        ...
    
    @property
    def device(self) -> device:
        """
        :obj:`torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        ...
    
    @property
    def dtype(self) -> torch.dtype:
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        ...
    
    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.

        Returns:
            :obj:`torch.Tensor`: The inverted attention mask.
        """
        ...
    
    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        ...
    
    def get_head_mask(self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = ...) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
        ...
    
    def num_parameters(self, only_trainable: bool = ..., exclude_embeddings: bool = ...) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            :obj:`int`: The number of parameters.
        """
        ...
    
    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (:obj:`dict`): The model inputs.

        Returns:
            :obj:`int`: The total number of tokens.
        """
        ...
    
    def floating_point_ops(self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = ...) -> int:
        """
        Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
        batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
        tokens (valid if :obj:`12 * d_model << sequence_length`) as laid out in `this paper
        <https://arxiv.org/pdf/2001.08361.pdf>`__ section 2.1. Should be overridden for transformers with parameter
        re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

        Args:
            batch_size (:obj:`int`):
                The batch size for the forward pass.

            sequence_length (:obj:`int`):
                The number of tokens in each line of the batch.

            exclude_embeddings (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to count embedding and softmax operations.

        Returns:
            :obj:`int`: The number of floating-point operations.
        """
        ...
    


class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin):
    r"""
    Base class for all models.

    :class:`~transformers.PreTrainedModel` takes care of storing the configuration of the models and handles methods
    for loading, downloading and saving models as well as a few methods common to all models to:

        * resize the input embeddings,
        * prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** (:class:`~transformers.PretrainedConfig`) -- A subclass of
          :class:`~transformers.PretrainedConfig` to use as configuration class for this model architecture.
        - **load_tf_weights** (:obj:`Callable`) -- A python `method` for loading a TensorFlow checkpoint in a PyTorch
          model, taking as arguments:

            - **model** (:class:`~transformers.PreTrainedModel`) -- An instance of the model on which to load the
              TensorFlow checkpoint.
            - **config** (:class:`~transformers.PreTrainedConfig`) -- An instance of the configuration associated to
              the model.
            - **path** (:obj:`str`) -- A path to the TensorFlow checkpoint.

        - **base_model_prefix** (:obj:`str`) -- A string indicating the attribute associated to the base model in
          derived classes of the same architecture adding modules on top of the base model.
        - **is_parallelizable** (:obj:`bool`) -- A flag indicating whether this model supports model parallelization.
    """
    config_class = ...
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_load_unexpected = ...
    _keys_to_ignore_on_save = ...
    is_parallelizable = ...
    @property
    def dummy_inputs(self) -> Dict[str, torch.Tensor]:
        """
        :obj:`Dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.
        """
        ...
    
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs) -> None:
        ...
    
    @property
    def base_model(self) -> nn.Module:
        """
        :obj:`torch.nn.Module`: The main body of the model.
        """
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        ...
    
    def set_input_embeddings(self, value: nn.Module):
        """
        Set model's input embeddings.

        Args:
            value (:obj:`nn.Module`): A module mapping vocabulary to hidden states.
        """
        ...
    
    def get_output_embeddings(self) -> nn.Module:
        """
        Returns the model's output embeddings.

        Returns:
            :obj:`nn.Module`: A torch module mapping hidden states to vocabulary.
        """
        ...
    
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
        the weights instead.
        """
        ...
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = ...) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if :obj:`new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a :obj:`tie_weights()` method.

        Arguments:
            new_num_tokens (:obj:`int`, `optional`):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or :obj:`None`,
                just returns a pointer to the input tokens :obj:`torch.nn.Embedding` module of the model without doing
                anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        ...
    
    def init_weights(self):
        """
        If needed prunes and maybe initializes weights.
        """
        ...
    
    def prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (:obj:`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (:obj:`int`) and associated values being the list of
                heads to prune in said layer (list of :obj:`int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads
                0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        ...
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike[str]], save_config: bool = ..., state_dict: Optional[Dict[Any, Any]] = ..., save_function: Callable[..., Any] = ..., push_to_hub: bool = ..., **kwargs: Any) -> None:
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            save_config (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to save the config of the model. Useful when in distributed training like TPUs and need
                to call this function on all processes. In this case, set :obj:`save_config=True` only on the main
                process to avoid race conditions.
            state_dict (nested dictionary of :obj:`torch.Tensor`):
                The state dictionary of the model to save. Will default to :obj:`self.state_dict()`, but can be used to
                only save parts of the model or if special precautions need to be taken when recovering the state
                dictionary of a model (like when using model parallelism).
            save_function (:obj:`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace :obj:`torch.save` by another method.
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
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated). To
        train the model, you should first set it back in training mode with ``model.train()``.

        The warning `Weights from XXX not initialized from pretrained model` means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning `Weights from XXX not used in YYY` means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a `flax checkpoint file` in `.msgpack` format (e.g,
                      ``./flax_model/`` containing ``flax_model.msgpack``). In this case, ``from_flax`` should be set
                      to :obj:`True`.
                    - :obj:`None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments ``config`` and ``state_dict``).
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
            state_dict (:obj:`Dict[str, torch.Tensor]`, `optional`):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using
                :func:`~transformers.PreTrainedModel.save_pretrained` and
                :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.
            cache_dir (:obj:`Union[str, os.PathLike]`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                ``pretrained_model_name_or_path`` argument).
            from_flax (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Load the model weights from a Flax checkpoint save file (see docstring of
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
            output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            mirror(:obj:`str`, `optional`):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            _fast_init(:obj:`bool`, `optional`, defaults to `:obj:`True`):
                Whether or not to disable fast initialization.
            torch_dtype (:obj:`str` or :obj:`torch.dtype`, `optional`):
                Override the default ``torch.dtype`` and load the model under this dtype. If ``"auto"`` is passed the
                dtype will be automatically derived from the model's weights.

                .. warning::

                    One should only disable `_fast_init` to ensure backwards compatibility with
                    ``transformers.__version__ < 4.6.0`` for seeded model initialization. This argument will be removed
                    at the next major version. See `pull request 11471
                    <https://github.com/huggingface/transformers/pull/11471>`__ for more information.

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

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.

        .. note::

            Activate the special `"offline-mode"
            <https://huggingface.co/transformers/installation.html#offline-mode>`__ to use this method in a firewalled
            environment.

        Examples::

            >>> from transformers import BertConfig, BertModel
            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = BertModel.from_pretrained('bert-base-uncased')
            >>> # Model was saved using `save_pretrained('./test/saved_model/')` (for example purposes, not runnable).
            >>> model = BertModel.from_pretrained('./test/saved_model/')
            >>> # Update configuration during loading.
            >>> model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> assert model.config.output_attentions == True
            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
            >>> config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            >>> model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)
            >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
            >>> model = BertModel.from_pretrained('bert-base-uncased', from_flax=True)

        """
        ...
    
    def retrieve_modules_from_names(self, names, add_prefix=..., remove_prefix=...):
        ...
    


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """
    def __init__(self, nf, nx) -> None:
        ...
    
    def forward(self, x):
        ...
    


class PoolerStartLogits(nn.Module):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model.
    """
    def __init__(self, config: PretrainedConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.FloatTensor, p_mask: Optional[torch.FloatTensor] = ...) -> torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            :obj:`torch.FloatTensor`: The start logits for SQuAD.
        """
        ...
    


class PoolerEndLogits(nn.Module):
    """
    Compute SQuAD end logits from sequence hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model and the
            :obj:`layer_norm_eps` to use.
    """
    def __init__(self, config: PretrainedConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.FloatTensor, start_states: Optional[torch.FloatTensor] = ..., start_positions: Optional[torch.LongTensor] = ..., p_mask: Optional[torch.FloatTensor] = ...) -> torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`, `optional`):
                The hidden states of the first tokens for the labeled span.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                The position of the first token for the labeled span.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        .. note::

            One of ``start_states`` or ``start_positions`` should be not obj:`None`. If both are set,
            ``start_positions`` overrides ``start_states``.

        Returns:
            :obj:`torch.FloatTensor`: The end logits for SQuAD.
        """
        ...
    


class PoolerAnswerClass(nn.Module):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.FloatTensor, start_states: Optional[torch.FloatTensor] = ..., start_positions: Optional[torch.LongTensor] = ..., cls_index: Optional[torch.LongTensor] = ...) -> torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`, `optional`):
                The hidden states of the first tokens for the labeled span.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                The position of the first token for the labeled span.
            cls_index (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Position of the CLS token for each sentence in the batch. If :obj:`None`, takes the last token.

        .. note::

            One of ``start_states`` or ``start_positions`` should be not obj:`None`. If both are set,
            ``start_positions`` overrides ``start_states``.

        Returns:
            :obj:`torch.FloatTensor`: The SQuAD 2.0 answer class.
        """
        ...
    


@dataclass
class SquadHeadOutput(ModelOutput):
    """
    Base class for outputs of question answering models using a :class:`~transformers.modeling_utils.SQuADHead`.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned if both :obj:`start_positions` and :obj:`end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (``torch.LongTensor`` of shape ``(batch_size, config.start_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (``torch.FloatTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Log probabilities for the top ``config.start_n_top * config.end_n_top`` end token possibilities
            (beam-search).
        end_top_index (``torch.LongTensor`` of shape ``(batch_size, config.start_n_top * config.end_n_top)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Indices for the top ``config.start_n_top * config.end_n_top`` end token possibilities (beam-search).
        cls_logits (``torch.FloatTensor`` of shape ``(batch_size,)``, `optional`, returned if ``start_positions`` or ``end_positions`` is not provided):
            Log probabilities for the ``is_impossible`` label of the answers.

    """
    loss: Optional[torch.FloatTensor] = ...
    start_top_log_probs: Optional[torch.FloatTensor] = ...
    start_top_index: Optional[torch.LongTensor] = ...
    end_top_log_probs: Optional[torch.FloatTensor] = ...
    end_top_index: Optional[torch.LongTensor] = ...
    cls_logits: Optional[torch.FloatTensor] = ...


class SQuADHead(nn.Module):
    r"""
    A SQuAD head inspired by XLNet.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model and the
            :obj:`layer_norm_eps` to use.
    """
    def __init__(self, config) -> None:
        ...
    
    @replace_return_docstrings(output_type=SquadHeadOutput, config_class=PretrainedConfig)
    def forward(self, hidden_states: torch.FloatTensor, start_positions: Optional[torch.LongTensor] = ..., end_positions: Optional[torch.LongTensor] = ..., cls_index: Optional[torch.LongTensor] = ..., is_impossible: Optional[torch.LongTensor] = ..., p_mask: Optional[torch.FloatTensor] = ..., return_dict: bool = ...) -> Union[SquadHeadOutput, Tuple[torch.FloatTensor]]:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                Final hidden states of the model on the sequence tokens.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Positions of the first token for the labeled span.
            end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Positions of the last token for the labeled span.
            cls_index (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Position of the CLS token for each sentence in the batch. If :obj:`None`, takes the last token.
            is_impossible (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Whether the question has a possible answer in the paragraph or not.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.
            return_dict (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.

        Returns:
        """
        ...
    


class SequenceSummary(nn.Module):
    r"""
    Compute a single vector summary of a sequence hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_type** (:obj:`str`) -- The method to use to make this summary. Accepted values are:

                - :obj:`"last"` -- Take the last token hidden state (like XLNet)
                - :obj:`"first"` -- Take the first token hidden state (like Bert)
                - :obj:`"mean"` -- Take the mean of all tokens hidden states
                - :obj:`"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
                - :obj:`"attn"` -- Not implemented now, use multi-head attention

            - **summary_use_proj** (:obj:`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (:obj:`bool`) -- If :obj:`True`, the projection outputs to
              :obj:`config.num_labels` classes (otherwise to :obj:`config.hidden_size`).
            - **summary_activation** (:obj:`Optional[str]`) -- Set to :obj:`"tanh"` to add a tanh activation to the
              output, another string or :obj:`None` will add no activation.
            - **summary_first_dropout** (:obj:`float`) -- Optional dropout probability before the projection and
              activation.
            - **summary_last_dropout** (:obj:`float`)-- Optional dropout probability after the projection and
              activation.
    """
    def __init__(self, config: PretrainedConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.FloatTensor, cls_index: Optional[torch.LongTensor] = ...) -> torch.FloatTensor:
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (:obj:`torch.LongTensor` of shape :obj:`[batch_size]` or :obj:`[batch_size, ...]` where ... are optional leading dimensions of :obj:`hidden_states`, `optional`):
                Used if :obj:`summary_type == "cls_index"` and takes the last token of the sequence as classification
                token.

        Returns:
            :obj:`torch.FloatTensor`: The summary of the sequence hidden states.
        """
        ...
    


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    ...

def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = ...) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (:obj:`torch.nn.Linear`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.

    Returns:
        :obj:`torch.nn.Linear`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    ...

def prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, dim: int = ...) -> Conv1D:
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.

    Used to remove heads.

    Args:
        layer (:class:`~transformers.modeling_utils.Conv1D`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 1): The dimension on which to keep the indices.

    Returns:
        :class:`~transformers.modeling_utils.Conv1D`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    ...

def prune_layer(layer: Union[nn.Linear, Conv1D], index: torch.LongTensor, dim: Optional[int] = ...) -> Union[nn.Linear, Conv1D]:
    """
    Prune a Conv1D or linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (:obj:`Union[torch.nn.Linear, Conv1D]`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`): The dimension on which to keep the indices.

    Returns:
        :obj:`torch.nn.Linear` or :class:`~transformers.modeling_utils.Conv1D`: The pruned layer as a new layer with
        :obj:`requires_grad=True`.
    """
    ...

def apply_chunking_to_forward(forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors) -> torch.Tensor:
    """
    This function chunks the :obj:`input_tensors` into smaller input tensor parts of size :obj:`chunk_size` over the
    dimension :obj:`chunk_dim`. It then applies a layer :obj:`forward_fn` to each chunk independently to save memory.

    If the :obj:`forward_fn` is independent across the :obj:`chunk_dim` this function will yield the same result as
    directly applying :obj:`forward_fn` to :obj:`input_tensors`.

    Args:
        forward_fn (:obj:`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (:obj:`int`):
            The chunk size of a chunked tensor: :obj:`num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (:obj:`int`):
            The dimension over which the :obj:`input_tensors` should be chunked.
        input_tensors (:obj:`Tuple[torch.Tensor]`):
            The input tensors of ``forward_fn`` which will be chunked

    Returns:
        :obj:`torch.Tensor`: A tensor with the same shape as the :obj:`forward_fn` would have given if applied`.


    Examples::

        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states

        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    """
    ...

