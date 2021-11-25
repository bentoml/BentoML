from typing import TYPE_CHECKING, Any, Dict

from . import dependency_versions_check
from .configuration_utils import PretrainedConfig
from .feature_extraction_utils import BatchFeature, SequenceFeatureExtractor
from .file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    SPIECE_UNDERLINE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    TensorType,
    _LazyModule,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
)
from .hf_argparser import HfArgumentParser
from .integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_ray_tune_available,
    is_tensorboard_available,
    is_wandb_available,
)
from .modeling_tf_pytorch_utils import (
    convert_tf_weight_name_to_pt_weight_name,
    load_pytorch_checkpoint_in_tf2_model,
    load_pytorch_model_in_tf2_model,
    load_pytorch_weights_in_tf2_model,
    load_tf2_checkpoint_in_pytorch_model,
    load_tf2_model_in_pytorch_model,
    load_tf2_weights_in_pytorch_model,
)
from .models.auto import (
    ALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CONFIG_MAPPING,
    FEATURE_EXTRACTOR_MAPPING,
    MODEL_NAMES_MAPPING,
    TOKENIZER_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
)
from .pipelines import (
    AutomaticSpeechRecognitionPipeline,
    Conversation,
    ConversationalPipeline,
    CsvPipelineDataFormat,
    FeatureExtractionPipeline,
    FillMaskPipeline,
    ImageClassificationPipeline,
    JsonPipelineDataFormat,
    NerPipeline,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    QuestionAnsweringPipeline,
    SummarizationPipeline,
    TableQuestionAnsweringPipeline,
    Text2TextGenerationPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
    TranslationPipeline,
    ZeroShotClassificationPipeline,
    pipeline,
)
from .tokenization_utils import PreTrainedTokenizer
from .tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    CharSpan,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TokenSpan,
)

__version__: str = ...
_import_structure: Dict[str, Any] = ...
