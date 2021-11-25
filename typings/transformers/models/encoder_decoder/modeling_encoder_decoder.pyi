

from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import Seq2SeqLMOutput
from ...modeling_utils import PreTrainedModel
from .configuration_encoder_decoder import EncoderDecoderConfig

""" Classes to support Encoder-Decoder architectures """
logger = ...
_CONFIG_FOR_DOC = ...
ENCODER_DECODER_START_DOCSTRING = ...
ENCODER_DECODER_INPUTS_DOCSTRING = ...
@add_start_docstrings(ENCODER_DECODER_START_DOCSTRING)
class EncoderDecoderModel(PreTrainedModel):
    r"""
    :class:`~transformers.EncoderDecoder` is a generic model class that will be instantiated as a transformer
    architecture with one of the base model classes of the library as encoder and another one as decoder when created
    with the :meth`~transformers.AutoModel.from_pretrained` class method for the encoder and
    :meth`~transformers.AutoModelForCausalLM.from_pretrained` class method for the decoder.
    """
    config_class = EncoderDecoderConfig
    base_model_prefix = ...
    def __init__(self, config: Optional[PretrainedConfig] = ..., encoder: Optional[PreTrainedModel] = ..., decoder: Optional[PreTrainedModel] = ...) -> None:
        ...
    
    def tie_weights(self): # -> None:
        ...
    
    def get_encoder(self): # -> PreTrainedModel | None:
        ...
    
    def get_decoder(self): # -> PreTrainedModel | None:
        ...
    
    def get_input_embeddings(self): # -> Module:
        ...
    
    def get_output_embeddings(self): # -> Module:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> Any:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs): # -> tuple[Unknown | EncoderDecoderModel, dict[str, Unbound | list[str] | list[Unknown]]] | EncoderDecoderModel:
        ...
    
    @classmethod
    def from_encoder_decoder_pretrained(cls, encoder_pretrained_model_name_or_path: str = ..., decoder_pretrained_model_name_or_path: str = ..., *model_args, **kwargs) -> PreTrainedModel:
        r"""
        Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using :obj:`model.eval()` (Dropout modules are deactivated). To
        train the model, you need to first set it back in training mode with :obj:`model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (:obj: `str`, `optional`):
                Information necessary to initiate the encoder. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.

            kwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                :obj:`output_attentions=True`).

                - To update the encoder configuration, use the prefix `encoder_` for each configuration parameter.
                - To update the decoder configuration, use the prefix `decoder_` for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a :obj:`config` is provided or automatically loaded.

        Example::

            >>> from transformers import EncoderDecoderModel
            >>> # initialize a bert2bert from two pretrained BERT models. Note that the cross-attention layers will be randomly initialized
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./bert2bert")
            >>> # load fine-tuned model
            >>> model = EncoderDecoderModel.from_pretrained("./bert2bert")

        """
        ...
    
    @add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., encoder_outputs=..., past_key_values=..., inputs_embeds=..., decoder_inputs_embeds=..., labels=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs): # -> Any | Seq2SeqLMOutput:
        r"""
        Returns:

        Examples::

            >>> from transformers import EncoderDecoderModel, BertTokenizer
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints

            >>> # forward
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

            >>> # training
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
            >>> loss, logits = outputs.loss, outputs.logits

            >>> # save and load from pretrained
            >>> model.save_pretrained("bert2bert")
            >>> model = EncoderDecoderModel.from_pretrained("bert2bert")

            >>> # generation
            >>> generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)

        """
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past=..., attention_mask=..., use_cache=..., encoder_outputs=..., **kwargs): # -> dict[str, Unknown | Any | None]:
        ...
    
    def resize_token_embeddings(self, *args, **kwargs):
        ...
    


