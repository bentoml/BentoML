

from ...configuration_utils import PretrainedConfig
from ...file_utils import add_start_docstrings

RAG_CONFIG_DOC = ...
@add_start_docstrings(RAG_CONFIG_DOC)
class RagConfig(PretrainedConfig):
    model_type = ...
    is_composition = ...
    def __init__(self, vocab_size=..., is_encoder_decoder=..., prefix=..., bos_token_id=..., pad_token_id=..., eos_token_id=..., decoder_start_token_id=..., title_sep=..., doc_sep=..., n_docs=..., max_combined_length=..., retrieval_vector_size=..., retrieval_batch_size=..., dataset=..., dataset_split=..., index_name=..., index_path=..., passages_path=..., use_dummy_dataset=..., reduce_loss=..., label_smoothing=..., do_deduplication=..., exclude_bos_score=..., do_marginalize=..., output_retrieved=..., use_cache=..., forced_eos_token_id=..., **kwargs) -> None:
        ...
    
    @classmethod
    def from_question_encoder_generator_configs(cls, question_encoder_config: PretrainedConfig, generator_config: PretrainedConfig, **kwargs) -> PretrainedConfig:
        r"""
        Instantiate a :class:`~transformers.EncoderDecoderConfig` (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            :class:`EncoderDecoderConfig`: An instance of a configuration object
        """
        ...
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default
        :meth:`~transformers.PretrainedConfig.to_dict`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        ...
    


