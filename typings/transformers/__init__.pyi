

from typing import TYPE_CHECKING, Any, Dict

from . import dependency_versions_check
from .configuration_utils import PretrainedConfig
from .data import (
    DataProcessor,
    InputExample,
    InputFeatures,
    SingleSentenceClassificationProcessor,
    SquadExample,
    SquadFeatures,
    SquadV1Processor,
    SquadV2Processor,
    glue_compute_metrics,
    glue_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
    squad_convert_examples_to_features,
    xnli_compute_metrics,
    xnli_output_modes,
    xnli_processors,
    xnli_tasks_num_labels,
)
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
    is_apex_available,
    is_datasets_available,
    is_faiss_available,
    is_flax_available,
    is_psutil_available,
    is_py3nvml_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_speech_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_tpu_available,
    is_vision_available,
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
from .modelcard import ModelCard
from .modeling_tf_pytorch_utils import (
    convert_tf_weight_name_to_pt_weight_name,
    load_pytorch_checkpoint_in_tf2_model,
    load_pytorch_model_in_tf2_model,
    load_pytorch_weights_in_tf2_model,
    load_tf2_checkpoint_in_pytorch_model,
    load_tf2_model_in_pytorch_model,
    load_tf2_weights_in_pytorch_model,
)
from .models.albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
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
from .models.bart import BartConfig, BartTokenizer
from .models.bert import (
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BasicTokenizer,
    BertConfig,
    BertTokenizer,
    WordpieceTokenizer,
)
from .models.bert_generation import BertGenerationConfig
from .models.bert_japanese import (
    BertJapaneseTokenizer,
    CharacterTokenizer,
    MecabTokenizer,
)
from .models.bertweet import BertweetTokenizer
from .models.big_bird import (
    BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BigBirdConfig,
    BigBirdTokenizer,
)
from .models.bigbird_pegasus import (
    BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BigBirdPegasusConfig,
)
from .models.blenderbot import (
    BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BlenderbotConfig,
    BlenderbotTokenizer,
)
from .models.blenderbot_small import (
    BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BlenderbotSmallConfig,
    BlenderbotSmallTokenizer,
)
from .models.byt5 import ByT5Tokenizer
from .models.camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from .models.canine import (
    CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CanineConfig,
    CanineTokenizer,
)
from .models.clip import (
    CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
    CLIPConfig,
    CLIPTextConfig,
    CLIPTokenizer,
    CLIPVisionConfig,
)
from .models.convbert import (
    CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ConvBertConfig,
    ConvBertTokenizer,
)
from .models.cpm import CpmTokenizer
from .models.ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig, CTRLTokenizer
from .models.deberta import (
    DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DebertaConfig,
    DebertaTokenizer,
)
from .models.deberta_v2 import DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaV2Config
from .models.deit import DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP, DeiTConfig
from .models.detr import DETR_PRETRAINED_CONFIG_ARCHIVE_MAP, DetrConfig
from .models.distilbert import (
    DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DistilBertConfig,
    DistilBertTokenizer,
)
from .models.dpr import (
    DPR_PRETRAINED_CONFIG_ARCHIVE_MAP,
    DPRConfig,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoderTokenizer,
    DPRReaderOutput,
    DPRReaderTokenizer,
)
from .models.electra import (
    ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ElectraConfig,
    ElectraTokenizer,
)
from .models.encoder_decoder import EncoderDecoderConfig
from .models.flaubert import (
    FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    FlaubertConfig,
    FlaubertTokenizer,
)
from .models.fsmt import FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, FSMTConfig, FSMTTokenizer
from .models.funnel import (
    FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    FunnelConfig,
    FunnelTokenizer,
)
from .models.gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2Tokenizer
from .models.gpt_neo import GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoConfig
from .models.herbert import HerbertTokenizer
from .models.hubert import HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, HubertConfig
from .models.ibert import IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, IBertConfig
from .models.layoutlm import (
    LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
    LayoutLMConfig,
    LayoutLMTokenizer,
)
from .models.led import LED_PRETRAINED_CONFIG_ARCHIVE_MAP, LEDConfig, LEDTokenizer
from .models.longformer import (
    LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
    LongformerConfig,
    LongformerTokenizer,
)
from .models.luke import LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP, LukeConfig, LukeTokenizer
from .models.lxmert import (
    LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    LxmertConfig,
    LxmertTokenizer,
)
from .models.m2m_100 import M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP, M2M100Config
from .models.marian import MarianConfig
from .models.mbart import MBartConfig
from .models.megatron_bert import (
    MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    MegatronBertConfig,
)
from .models.mmbt import MMBTConfig
from .models.mobilebert import (
    MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    MobileBertConfig,
    MobileBertTokenizer,
)
from .models.mpnet import (
    MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    MPNetConfig,
    MPNetTokenizer,
)
from .models.mt5 import MT5Config
from .models.openai import (
    OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    OpenAIGPTConfig,
    OpenAIGPTTokenizer,
)
from .models.pegasus import PegasusConfig
from .models.phobert import PhobertTokenizer
from .models.prophetnet import (
    PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ProphetNetConfig,
    ProphetNetTokenizer,
)
from .models.rag import RagConfig, RagRetriever, RagTokenizer
from .models.reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig
from .models.retribert import (
    RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    RetriBertConfig,
    RetriBertTokenizer,
)
from .models.roberta import (
    ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    RobertaConfig,
    RobertaTokenizer,
)
from .models.roformer import (
    ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
    RoFormerConfig,
    RoFormerTokenizer,
)
from .models.speech_to_text import (
    SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    Speech2TextConfig,
)
from .models.squeezebert import (
    SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    SqueezeBertConfig,
    SqueezeBertTokenizer,
)
from .models.t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
from .models.tapas import (
    TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP,
    TapasConfig,
    TapasTokenizer,
)
from .models.transfo_xl import (
    TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    TransfoXLConfig,
    TransfoXLCorpus,
    TransfoXLTokenizer,
)
from .models.visual_bert import (
    VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    VisualBertConfig,
)
from .models.vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig
from .models.wav2vec2 import (
    WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2Tokenizer,
)
from .models.xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig, XLMTokenizer
from .models.xlm_prophetnet import (
    XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLMProphetNetConfig,
)
from .models.xlm_roberta import (
    XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLMRobertaConfig,
)
from .models.xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig
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
from .trainer_callback import (
    DefaultFlowCallback,
    EarlyStoppingCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from .trainer_utils import EvalPrediction, IntervalStrategy, SchedulerType, set_seed
from .training_args import TrainingArguments
from .training_args_seq2seq import Seq2SeqTrainingArguments
from .training_args_tf import TFTrainingArguments
from .utils import (
    dummy_flax_objects,
    dummy_pt_objects,
    dummy_sentencepiece_and_speech_objects,
    dummy_sentencepiece_and_tokenizers_objects,
    dummy_sentencepiece_objects,
    dummy_speech_objects,
    dummy_tf_objects,
    dummy_timm_objects,
    dummy_tokenizers_objects,
    dummy_vision_objects,
    logging,
)

__version__: str = ...
_import_structure: Dict[str, Any] = ...
