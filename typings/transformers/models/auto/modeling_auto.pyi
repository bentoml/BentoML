from .auto_factory import _BaseAutoModelClass

logger = ...
MODEL_MAPPING = ...
MODEL_FOR_PRETRAINING_MAPPING = ...
MODEL_WITH_LM_HEAD_MAPPING = ...
MODEL_FOR_CAUSAL_LM_MAPPING = ...
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = ...
MODEL_FOR_MASKED_LM_MAPPING = ...
MODEL_FOR_OBJECT_DETECTION_MAPPING = ...
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = ...
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = ...
MODEL_FOR_QUESTION_ANSWERING_MAPPING = ...
MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = ...
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = ...
MODEL_FOR_MULTIPLE_CHOICE_MAPPING = ...
MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = ...

class AutoModel(_BaseAutoModelClass):
    _model_mapping = ...

AutoModel = ...

class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = ...

AutoModelForPreTraining = ...

class _AutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = ...

_AutoModelWithLMHead = ...

class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = ...

AutoModelForCausalLM = ...

class AutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = ...

AutoModelForMaskedLM = ...

class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = ...

AutoModelForSeq2SeqLM = ...

class AutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = ...

AutoModelForSequenceClassification = ...

class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = ...

AutoModelForQuestionAnswering = ...

class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = ...

AutoModelForTableQuestionAnswering = ...

class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = ...

AutoModelForTokenClassification = ...

class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = ...

AutoModelForMultipleChoice = ...

class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = ...

AutoModelForNextSentencePrediction = ...

class AutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = ...

AutoModelForImageClassification = ...

class AutoModelWithLMHead(_AutoModelWithLMHead):
    @classmethod
    def from_config(cls, config): ...
