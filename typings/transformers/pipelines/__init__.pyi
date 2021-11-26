import warnings
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple, Union, overload
import tensorflow as tf
from transformers.utils.dummy_flax_objects import FlaxPreTrainedModel
from ..configuration_utils import PretrainedConfig
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..modeling_tf_utils import TFPreTrainedModel
from ..modeling_utils import PreTrainedModel
from ..models.auto.configuration_auto import AutoConfig
from ..models.auto.feature_extraction_auto import (
    FEATURE_EXTRACTOR_MAPPING,
    AutoFeatureExtractor,
)
from ..models.auto.modeling_auto import (
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTableQuestionAnswering,
    AutoModelForTokenClassification,
)
from ..models.auto.modeling_tf_auto import (
    TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    TF_MODEL_WITH_LM_HEAD_MAPPING,
    TFAutoModel,
    TFAutoModelForCausalLM,
    TFAutoModelForMaskedLM,
    TFAutoModelForQuestionAnswering,
    TFAutoModelForSeq2SeqLM,
    TFAutoModelForSequenceClassification,
    TFAutoModelForTokenClassification,
)
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import logging
from .automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from .base import (
    ArgumentHandler,
    CsvPipelineDataFormat,
    JsonPipelineDataFormat,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    PipelineException,
    get_default_model,
    infer_framework_load_model,
)
from .conversational import Conversation, ConversationalPipeline
from .feature_extraction import FeatureExtractionPipeline
from .fill_mask import FillMaskPipeline
from .image_classification import ImageClassificationPipeline
from .question_answering import (
    QuestionAnsweringArgumentHandler,
    QuestionAnsweringPipeline,
)
from .table_question_answering import (
    TableQuestionAnsweringArgumentHandler,
    TableQuestionAnsweringPipeline,
)
from .text2text_generation import (
    SummarizationPipeline,
    Text2TextGenerationPipeline,
    TranslationPipeline,
)
from .text_classification import TextClassificationPipeline
from .text_generation import TextGenerationPipeline
from .token_classification import (
    AggregationStrategy,
    NerPipeline,
    TokenClassificationArgumentHandler,
    TokenClassificationPipeline,
)
from .zero_shot_classification import (
    ZeroShotClassificationArgumentHandler,
    ZeroShotClassificationPipeline,
)

TASK_ALIASES: Dict[str, str] = ...
SUPPORTED_TASKS: Mapping[str, Any] = ...

def check_task(task: str) -> Tuple[Dict[str, Any], Any]: ...
def pipeline(
    task: str,
    model: Optional[
        Union[str, FlaxPreTrainedModel, PreTrainedModel, TFPreTrainedModel]
    ] = ...,
    config: Optional[Union[str, PretrainedConfig]] = ...,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = ...,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = ...,
    framework: Optional[str] = ...,
    revision: Optional[str] = ...,
    use_fast: bool = ...,
    use_auth_token: Optional[Union[str, bool]] = ...,
    model_kwargs: Dict[str, Any] = ...,
    **kwargs: Any
) -> Pipeline: ...
