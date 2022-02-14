import typing as t

from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines.base import Pipeline as TransformersPipeline
from transformers.modeling_tf_utils import TFPreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.feature_extraction_sequence_utils import (
    SequenceFeatureExtractor as PreTrainedFeatureExtractor,
)

TransformersModelType = t.Union[PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel]
TransformersTokenizerType = t.Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

__all__ = [
    "TransformersPipeline",
    "TransformersModelType",
    "TransformersTokenizerType",
    "PreTrainedFeatureExtractor",
    "PretrainedConfig",
]
