

from ...configuration_utils import PretrainedConfig

logger = ...
class EncoderDecoderConfig(PretrainedConfig):
    r"""
    :class:`~transformers.EncoderDecoderConfig` is the configuration class to store the configuration of a
    :class:`~transformers.EncoderDecoderModel`. It is used to instantiate an Encoder Decoder model according to the
    specified arguments, defining the encoder and decoder configs.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        kwargs (`optional`):
            Dictionary of keyword arguments. Notably:

                - **encoder** (:class:`~transformers.PretrainedConfig`, `optional`) -- An instance of a configuration
                  object that defines the encoder config.
                - **decoder** (:class:`~transformers.PretrainedConfig`, `optional`) -- An instance of a configuration
                  object that defines the decoder config.

    Examples::

        >>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

        >>> # Initializing a BERT bert-base-uncased style configuration
        >>> config_encoder = BertConfig()
        >>> config_decoder = BertConfig()

        >>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

        >>> # Initializing a Bert2Bert model from the bert-base-uncased style configurations
        >>> model = EncoderDecoderModel(config=config)

        >>> # Accessing the model configuration
        >>> config_encoder = model.config.encoder
        >>> config_decoder  = model.config.decoder
        >>> # set decoder config to causal lm
        >>> config_decoder.is_decoder = True
        >>> config_decoder.add_cross_attention = True

        >>> # Saving the model, including its configuration
        >>> model.save_pretrained('my-model')

        >>> # loading model and config from pretrained folder
        >>> encoder_decoder_config = EncoderDecoderConfig.from_pretrained('my-model')
        >>> model = EncoderDecoderModel.from_pretrained('my-model', config=encoder_decoder_config)
    """
    model_type = ...
    is_composition = ...
    def __init__(self, **kwargs) -> None:
        ...
    
    @classmethod
    def from_encoder_decoder_configs(cls, encoder_config: PretrainedConfig, decoder_config: PretrainedConfig, **kwargs) -> PretrainedConfig:
        r"""
        Instantiate a :class:`~transformers.EncoderDecoderConfig` (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            :class:`EncoderDecoderConfig`: An instance of a configuration object
        """
        ...
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        ...
    


