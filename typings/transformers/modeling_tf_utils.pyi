
import os
from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from .configuration_utils import PretrainedConfig
from .file_utils import PushToHubMixin
from .generation_tf_utils import TFGenerationMixin

logger = ...
tf_logger = ...
TFModelInputType = Union[List[tf.Tensor], List[np.ndarray], Dict[str, tf.Tensor], Dict[str, np.ndarray], np.ndarray, tf.Tensor]
class TFModelUtilsMixin:
    """
    A few utilities for :obj:`tf.keras.Model`, to be used as a mixin.
    """
    def num_parameters(self, only_trainable: bool = ...) -> int:
        """
        Get the number of (optionally, trainable) parameters in the model.

        Args:
            only_trainable (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return only the number of trainable parameters

        Returns:
            :obj:`int`: The number of parameters.
        """
        ...
    


def keras_serializable(cls):
    """
    Decorate a Keras Layer class to support Keras serialization.

    This is done by:

    1. Adding a :obj:`transformers_config` dict to the Keras config dictionary in :obj:`get_config` (called by Keras at
       serialization time.
    2. Wrapping :obj:`__init__` to accept that :obj:`transformers_config` dict (passed by Keras at deserialization
       time) and convert it to a config object for the actual layer initializer.
    3. Registering the class as a custom object in Keras (if the Tensorflow version supports this), so that it does not
       need to be supplied in :obj:`custom_objects` in the call to :obj:`tf.keras.models.load_model`.

    Args:
        cls (a :obj:`tf.keras.layers.Layers subclass`):
            Typically a :obj:`TF.MainLayer` class in this project, in general must accept a :obj:`config` argument to
            its initializer.

    Returns:
        The same class object, with modifications for Keras deserialization.
    """
    ...

class TFCausalLanguageModelingLoss:
    """
    Loss function suitable for causal language modeling (CLM), that is, the task of guessing the next token.

    .. note::

        Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    """
    def compute_loss(self, labels, logits):
        ...
    


class TFQuestionAnsweringLoss:
    """
    Loss function suitable for question answering.
    """
    def compute_loss(self, labels, logits):
        ...
    


class TFTokenClassificationLoss:
    """
    Loss function suitable for token classification.

    .. note::

        Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    """
    def compute_loss(self, labels, logits):
        ...
    


class TFSequenceClassificationLoss:
    """
    Loss function suitable for sequence classification.
    """
    def compute_loss(self, labels, logits):
        ...
    


class TFMultipleChoiceLoss(TFSequenceClassificationLoss):
    """Loss function suitable for multiple choice tasks."""
    ...


class TFMaskedLanguageModelingLoss(TFCausalLanguageModelingLoss):
    """
    Loss function suitable for masked language modeling (MLM), that is, the task of guessing the masked tokens.

    .. note::

         Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.
    """
    ...


class TFNextSentencePredictionLoss:
    """
    Loss function suitable for next sentence prediction (NSP), that is, the task of guessing the next sentence.

    .. note::
         Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.
    """
    def compute_loss(self, labels, logits):
        ...
    


def booleans_processing(config, **kwargs):
    """
    Process the input booleans of each model in order to be sure they are compliant with the execution mode (eager or
    graph)

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config of the running model.
        **kwargs:
            The boolean parameters

    Returns:
        A dictionary with the proper values for each boolean
    """
    ...

def input_processing(func, config, input_ids, **kwargs):
    """
    Process the input of each TensorFlow model including the booleans. In case of a list of symbolic inputs, each input
    has to be named accordingly to the parameters name, i.e. `input_ids = tf.keras.Input(shape=(128,), dtype='int32',
    name="input_ids")` otherwise the order of the tensors will not be guaranteed during the training.

    Args:
        func (:obj:`callable`):
            The callable function of the TensorFlow model.
        config (:class:`~transformers.PretrainedConfig`):
            The config of the running model.
        **kwargs:
            The inputs of the model.

    Returns:
        Two lists, one for the missing layers, and another one for the unexpected layers.
    """
    ...

def load_tf_weights(model, resolved_archive_file, ignore_mismatched_sizes=..., _prefix=...):
    """
    Detect missing and unexpected layers and load the TF weights accordingly to their names and shapes.

    Args:
        model (:obj:`tf.keras.models.Model`):
            The model to load the weights into.
        resolved_archive_file (:obj:`str`):
            The location of the H5 file.
        ignore_mismatched_sizes (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to ignore weights with shapes that don't match between the checkpoint of the model.

    Returns:
        Three lists, one for the missing layers, another one for the unexpected layers, and a last one for the
        mismatched layers.
    """
    ...

def init_copy_embeddings(old_embeddings, new_num_tokens):
    r"""
    This function aims to reduce the embeddings in case new_num_tokens < old_num_tokens or to pad with -1 in case
    new_num_tokens > old_num_tokens. A mask is also computed in order to know which weight in the embeddings should be
    kept or not. Example:

        - if new_num_tokens=5 and old_num_tokens=4 and old_embeddings=[w1,w2,w3,w4]

            -  mask=[True,True,True,True,False] and current_weights=[w1,w2,w3,w4,-1]
        - if new_num_tokens=4 and old_num_tokens=5 and old_embeddings=[w1,w2,w3,w4,w5]

            - mask=[True,True,True,True] and current_weights=[w1,w2,w3,w4]
    """
    ...

class TFPreTrainedModel(tf.keras.Model, TFModelUtilsMixin, TFGenerationMixin, PushToHubMixin):
    r"""
    Base class for all TF models.

    :class:`~transformers.TFPreTrainedModel` takes care of storing the configuration of the models and handles methods
    for loading, downloading and saving models as well as a few methods common to all models to:

        * resize the input embeddings,
        * prune heads in the self-attention heads.

    Class attributes (overridden by derived classes):

        - **config_class** (:class:`~transformers.PretrainedConfig`) -- A subclass of
          :class:`~transformers.PretrainedConfig` to use as configuration class for this model architecture.
        - **base_model_prefix** (:obj:`str`) -- A string indicating the attribute associated to the base model in
          derived classes of the same architecture adding modules on top of the base model.
    """
    config_class = ...
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _keys_to_ignore_on_load_unexpected = ...
    _requires_load_weight_prefix = ...
    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            :obj:`Dict[str, tf.Tensor]`: The dummy inputs.
        """
        ...
    
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),"token_type_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids") }])
    def serving(self, inputs):
        """
        Method used for serving the model.

        Args:
            inputs (:obj:`Dict[str, tf.Tensor]`):
                The input of the saved model as a dictionary of tensors.
        """
        ...
    
    def serving_output(output):
        """
        Prepare the output of the saved model. Each model must implement this function.

        Args:
            output (:obj:`~transformers.TFBaseModelOutput`):
                The output returned by the model.
        """
        ...
    
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        """
        Returns the model's input embeddings layer.

        Returns:
            :obj:`tf.Variable`: The embeddings layer mapping vocabulary to hidden states.
        """
        ...
    
    def set_input_embeddings(self, value):
        """
        Set model's input embeddings

        Args:
            value (:obj:`tf.Variable`):
                The new weights mapping hidden states to vocabulary.
        """
        ...
    
    def get_output_embeddings(self) -> Union[None, tf.keras.layers.Layer]:
        """
        Returns the model's output embeddings

        Returns:
            :obj:`tf.Variable`: The new weights mapping vocabulary to hidden states.
        """
        ...
    
    def set_output_embeddings(self, value):
        """
        Set model's output embeddings

        Args:
            value (:obj:`tf.Variable`):
                The new weights mapping hidden states to vocabulary.
        """
        ...
    
    def get_output_layer_with_bias(self) -> Union[None, tf.keras.layers.Layer]:
        """
        Get the layer that handles a bias attribute in case the model has an LM head with weights tied to the
        embeddings

        Return:
            :obj:`tf.keras.layers.Layer`: The layer that handles the bias, None if not an LM model.
        """
        ...
    
    def get_prefix_bias_name(self) -> Union[None, str]:
        """
        Get the concatenated _prefix name of the bias from the model name to the parent layer

        Return:
            :obj:`str`: The _prefix name of the bias.
        """
        ...
    
    def get_bias(self) -> Union[None, Dict[str, tf.Variable]]:
        """
        Dict of bias attached to an LM head. The key represents the name of the bias attribute.

        Return:
            :obj:`tf.Variable`: The weights representing the bias, None if not an LM model.
        """
        ...
    
    def set_bias(self, value):
        """
        Set all the bias in the LM head.

        Args:
            value (:obj:`Dict[tf.Variable]`):
                All the new bias attached to an LM head.
        """
        ...
    
    def get_lm_head(self) -> tf.keras.layers.Layer:
        """
        The LM Head layer. This method must be overwritten by all the models that have a lm head.

        Return:
            :obj:`tf.keras.layers.Layer`: The LM head layer if the model has one, None if not.
        """
        ...
    
    def resize_token_embeddings(self, new_num_tokens=...) -> tf.Variable:
        """
        Resizes input token embeddings matrix of the model if :obj:`new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a :obj:`tie_weights()` method.

        Arguments:
            new_num_tokens (:obj:`int`, `optional`):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or :obj:`None`,
                just returns a pointer to the input tokens :obj:`tf.Variable` module of the model without doing
                anything.

        Return:
            :obj:`tf.Variable`: Pointer to the input tokens Embeddings Module of the model.
        """
        ...
    
    def prune_heads(self, heads_to_prune):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (:obj:`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (:obj:`int`) and associated values being the list of
                heads to prune in said layer (list of :obj:`int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads
                0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        ...
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike[str]], saved_model:bool =..., version: int=..., push_to_hub: bool=..., **kwargs: Any) -> None:
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        :func:`~transformers.TFPreTrainedModel.from_pretrained` class method.

        Arguments:
            save_directory (:obj:`str`):
                Directory to which to save. Will be created if it doesn't exist.
            saved_model (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If the model has to be saved in saved model format as well or not.
            version (:obj:`int`, `optional`, defaults to 1):
                The version of the saved model. A saved model needs to be versioned in order to be properly loaded by
                TensorFlow Serving as detailed in the official documentation
                https://www.tensorflow.org/tfx/serving/serving_basic
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
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiate a pretrained TF 2.0 model from a pre-trained model configuration.

        The warning `Weights from XXX not initialized from pretrained model` means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning `Weights from XXX not used in YYY` means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (:obj:`str`, `optional`):
                Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `PyTorch state_dict save file` (e.g, ``./pt_model/pytorch_model.bin``). In
                      this case, ``from_pt`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the PyTorch model in a
                      TensorFlow model using the provided conversion scripts and loading the TensorFlow model
                      afterwards.
                    - :obj:`None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments ``config`` and ``state_dict``).
            model_args (sequence of positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
            config (:obj:`Union[PretrainedConfig, str]`, `optional`):
                Can be either:

                    - an instance of a class derived from :class:`~transformers.PretrainedConfig`,
                    - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`.

                Configuration for the model to use instead of an automatically loaded configuation. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the `model id` string of a pretrained
                      model).
                    - The model was saved using :func:`~transformers.TFPreTrainedModel.save_pretrained` and is reloaded
                      by supplying the save directory.
                    - The model is loaded by supplying a local directory as ``pretrained_model_name_or_path`` and a
                      configuration JSON file named `config.json` is found in the directory.
            from_pt: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Load the model weights from a PyTorch state_dict save file (see docstring of
                ``pretrained_model_name_or_path`` argument).
            ignore_mismatched_sizes (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies: (:obj:`Dict[str, str], `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to only look at local files (e.g., not try doanloading the model).
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

        Examples::

            >>> from transformers import BertConfig, TFBertModel
            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = TFBertModel.from_pretrained('bert-base-uncased')
            >>> # Model was saved using `save_pretrained('./test/saved_model/')` (for example purposes, not runnable).
            >>> model = TFBertModel.from_pretrained('./test/saved_model/')
            >>> # Update configuration during loading.
            >>> model = TFBertModel.from_pretrained('bert-base-uncased', output_attentions=True)
            >>> assert model.config.output_attentions == True
            >>> # Loading from a Pytorch model file instead of a TensorFlow checkpoint (slower, for example purposes, not runnable).
            >>> config = BertConfig.from_json_file('./pt_model/my_pt_model_config.json')
            >>> model = TFBertModel.from_pretrained('./pt_model/my_pytorch_model.bin', from_pt=True, config=config)

        """
        ...
    


class TFConv1D(tf.keras.layers.Layer):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`):
            The number of output features.
        nx (:obj:`int`):
            The number of input features.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation to use to initialize the weights.
        kwargs:
            Additional keyword arguments passed along to the :obj:`__init__` of :obj:`tf.keras.layers.Layer`.
    """
    def __init__(self, nf, nx, initializer_range=..., **kwargs) -> None:
        ...
    
    def build(self, input_shape):
        ...
    
    def call(self, x):
        ...
    


class TFSharedEmbeddings(tf.keras.layers.Layer):
    r"""
    Construct shared token embeddings.

    The weights of the embedding layer is usually shared with the weights of the linear decoder when doing language
    modeling.

    Args:
        vocab_size (:obj:`int`):
            The size of the vocabulary, e.g., the number of unique tokens.
        hidden_size (:obj:`int`):
            The size of the embedding vectors.
        initializer_range (:obj:`float`, `optional`):
            The standard deviation to use when initializing the weights. If no value is provided, it will default to
            :math:`1/\sqrt{hidden\_size}`.
        kwargs:
            Additional keyword arguments passed along to the :obj:`__init__` of :obj:`tf.keras.layers.Layer`.
    """
    def __init__(self, vocab_size: int, hidden_size: int, initializer_range: Optional[float] = ..., **kwargs) -> None:
        ...
    
    def build(self, input_shape):
        """
        Build shared token embedding layer Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        ...
    
    def get_config(self):
        ...
    
    def call(self, inputs: tf.Tensor, mode: str = ...) -> tf.Tensor:
        """
        Get token embeddings of inputs or decode final hidden state.

        Args:
            inputs (:obj:`tf.Tensor`):
                In embedding mode, should be an int64 tensor with shape :obj:`[batch_size, length]`.

                In linear mode, should be a float tensor with shape :obj:`[batch_size, length, hidden_size]`.
            mode (:obj:`str`, defaults to :obj:`"embedding"`):
               A valid value is either :obj:`"embedding"` or :obj:`"linear"`, the first one indicates that the layer
               should be used as an embedding layer, the second one that the layer should be used as a linear decoder.

        Returns:
            :obj:`tf.Tensor`: In embedding mode, the output is a float32 embedding tensor, with shape
            :obj:`[batch_size, length, embedding_size]`.

            In linear mode, the output is a float32 with shape :obj:`[batch_size, length, vocab_size]`.

        Raises:
            ValueError: if :obj:`mode` is not valid.

        Shared weights logic is adapted from `here
        <https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24>`__.
        """
        ...
    


class TFSequenceSummary(tf.keras.layers.Layer):
    """
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

        initializer_range (:obj:`float`, defaults to 0.02): The standard deviation to use to initialize the weights.
        kwargs:
            Additional keyword arguments passed along to the :obj:`__init__` of :obj:`tf.keras.layers.Layer`.
    """
    def __init__(self, config: PretrainedConfig, initializer_range: float = ..., **kwargs) -> None:
        ...
    
    def call(self, inputs, cls_index=..., training=...):
        ...
    


def shape_list(tensor: tf.Tensor) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (:obj:`tf.Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    ...

def get_initializer(initializer_range: float = ...) -> tf.initializers.TruncatedNormal:
    """
    Creates a :obj:`tf.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (`float`, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        :obj:`tf.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    ...

class TFWrappedEmbeddings:
    """
    this class wraps a the TFSharedEmbeddingTokens layer into a python 'no-keras-layer' class to avoid problem with
    weight restoring. Also it makes sure that the layer is called from the correct scope to avoid problem with
    saving/storing the correct weights
    """
    def __init__(self, layer, abs_scope_name=...) -> None:
        ...
    
    def call(self, inputs, mode=...):
        ...
    
    def __call__(self, inputs, mode=...):
        ...
    


