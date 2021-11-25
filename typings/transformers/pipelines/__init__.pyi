

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import tensorflow as tf
import torch

from ..configuration_utils import PretrainedConfig
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..file_utils import is_tf_available, is_torch_available
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

if is_tf_available():
    ...
if is_torch_available():
    ...
if TYPE_CHECKING:
    ...
logger = ...
TASK_ALIASES = ...
SUPPORTED_TASKS = ...
def check_task(task: str) -> Tuple[Dict, Any]:
    """
    Checks an incoming task string, to validate it's correct and return the default Pipeline and Model classes, and
    default models if they exist.

    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - :obj:`"feature-extraction"`
            - :obj:`"text-classification"`
            - :obj:`"sentiment-analysis"` (alias of :obj:`"text-classification")
            - :obj:`"token-classification"`
            - :obj:`"ner"` (alias of :obj:`"token-classification")
            - :obj:`"question-answering"`
            - :obj:`"fill-mask"`
            - :obj:`"summarization"`
            - :obj:`"translation_xx_to_yy"`
            - :obj:`"translation"`
            - :obj:`"text-generation"`
            - :obj:`"conversational"`

    Returns:
        (task_defaults:obj:`dict`, task_options: (:obj:`tuple`, None)) The actual dictionary required to initialize the
        pipeline and some extra task options for parametrized tasks like "translation_XX_to_YY"


    """
    ...

def pipeline(task: str, model: Optional = ..., config: Optional[Union[str, PretrainedConfig]] = ..., tokenizer: Optional[Union[str, PreTrainedTokenizer]] = ..., feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = ..., framework: Optional[str] = ..., revision: Optional[str] = ..., use_fast: bool = ..., use_auth_token: Optional[Union[str, bool]] = ..., model_kwargs: Dict[str, Any] = ..., **kwargs) -> Pipeline:
    """
    Utility factory method to build a :class:`~transformers.Pipeline`.

    Pipelines are made of:

        - A :doc:`tokenizer <tokenizer>` in charge of mapping raw textual input to token.
        - A :doc:`model <model>` to make predictions from the inputs.
        - Some (optional) post processing for enhancing model's output.

    Args:
        task (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:

            - :obj:`"feature-extraction"`: will return a :class:`~transformers.FeatureExtractionPipeline`.
            - :obj:`"text-classification"`: will return a :class:`~transformers.TextClassificationPipeline`.
            - :obj:`"sentiment-analysis"`: (alias of :obj:`"text-classification") will return a
              :class:`~transformers.TextClassificationPipeline`.
            - :obj:`"token-classification"`: will return a :class:`~transformers.TokenClassificationPipeline`.
            - :obj:`"ner"` (alias of :obj:`"token-classification"): will return a
              :class:`~transformers.TokenClassificationPipeline`.
            - :obj:`"question-answering"`: will return a :class:`~transformers.QuestionAnsweringPipeline`.
            - :obj:`"fill-mask"`: will return a :class:`~transformers.FillMaskPipeline`.
            - :obj:`"summarization"`: will return a :class:`~transformers.SummarizationPipeline`.
            - :obj:`"translation_xx_to_yy"`: will return a :class:`~transformers.TranslationPipeline`.
            - :obj:`"text2text-generation"`: will return a :class:`~transformers.Text2TextGenerationPipeline`.
            - :obj:`"text-generation"`: will return a :class:`~transformers.TextGenerationPipeline`.
            - :obj:`"zero-shot-classification:`: will return a :class:`~transformers.ZeroShotClassificationPipeline`.
            - :obj:`"conversational"`: will return a :class:`~transformers.ConversationalPipeline`.
        model (:obj:`str` or :obj:`~transformers.PreTrainedModel` or :obj:`~transformers.TFPreTrainedModel`, `optional`):
            The model that will be used by the pipeline to make predictions. This can be a model identifier or an
            actual instance of a pretrained model inheriting from :class:`~transformers.PreTrainedModel` (for PyTorch)
            or :class:`~transformers.TFPreTrainedModel` (for TensorFlow).

            If not provided, the default for the :obj:`task` will be loaded.
        config (:obj:`str` or :obj:`~transformers.PretrainedConfig`, `optional`):
            The configuration that will be used by the pipeline to instantiate the model. This can be a model
            identifier or an actual pretrained model configuration inheriting from
            :class:`~transformers.PretrainedConfig`.

            If not provided, the default configuration file for the requested model will be used. That means that if
            :obj:`model` is given, its default configuration will be used. However, if :obj:`model` is not supplied,
            this :obj:`task`'s default model's config is used instead.
        tokenizer (:obj:`str` or :obj:`~transformers.PreTrainedTokenizer`, `optional`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from :class:`~transformers.PreTrainedTokenizer`.

            If not provided, the default tokenizer for the given :obj:`model` will be loaded (if it is a string). If
            :obj:`model` is not specified or not a string, then the default tokenizer for :obj:`config` is loaded (if
            it is a string). However, if :obj:`config` is also not given or not a string, then the default tokenizer
            for the given :obj:`task` will be loaded.
        feature_extractor (:obj:`str` or :obj:`~transformers.PreTrainedFeatureExtractor`, `optional`):
            The feature extractor that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained feature extractor inheriting from
            :class:`~transformers.PreTrainedFeatureExtractor`.

            Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal
            models. Multi-modal models will also require a tokenizer to be passed.

            If not provided, the default feature extractor for the given :obj:`model` will be loaded (if it is a
            string). If :obj:`model` is not specified or not a string, then the default feature extractor for
            :obj:`config` is loaded (if it is a string). However, if :obj:`config` is also not given or not a string,
            then the default feature extractor for the given :obj:`task` will be loaded.
        framework (:obj:`str`, `optional`):
            The framework to use, either :obj:`"pt"` for PyTorch or :obj:`"tf"` for TensorFlow. The specified framework
            must be installed.

            If no framework is specified, will default to the one currently installed. If no framework is specified and
            both frameworks are installed, will default to the framework of the :obj:`model`, or to PyTorch if no model
            is provided.
        revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
            When passing a task name or a string model identifier: The specific model version to use. It can be a
            branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so ``revision`` can be any identifier allowed by git.
        use_fast (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use a Fast tokenizer if possible (a :class:`~transformers.PreTrainedTokenizerFast`).
        use_auth_token (:obj:`str` or `bool`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
        model_kwargs:
            Additional dictionary of keyword arguments passed along to the model's :obj:`from_pretrained(...,
            **model_kwargs)` function.
        kwargs:
            Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
            corresponding pipeline class for possible values).

    Returns:
        :class:`~transformers.Pipeline`: A suitable pipeline for the task.

    Examples::

        >>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

        >>> # Sentiment analysis pipeline
        >>> pipeline('sentiment-analysis')

        >>> # Question answering pipeline, specifying the checkpoint identifier
        >>> pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

        >>> # Named entity recognition pipeline, passing in a specific model and tokenizer
        >>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        >>> pipeline('ner', model=model, tokenizer=tokenizer)
    """
    ...

