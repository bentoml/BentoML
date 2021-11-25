

from .auto_factory import _BaseAutoModelClass

logger = ...
TF_MODEL_MAPPING = ...
TF_MODEL_FOR_PRETRAINING_MAPPING = ...
TF_MODEL_WITH_LM_HEAD_MAPPING = ...
TF_MODEL_FOR_CAUSAL_LM_MAPPING = ...
TF_MODEL_FOR_MASKED_LM_MAPPING = ...
TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = ...
TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = ...
TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING = ...
TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = ...
TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = ...
TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = ...
class TFAutoModel(_BaseAutoModelClass):
    _model_mapping = ...


TFAutoModel = ...
class TFAutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = ...


TFAutoModelForPreTraining = ...
class _TFAutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = ...


_TFAutoModelWithLMHead = ...
class TFAutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = ...


TFAutoModelForCausalLM = ...
class TFAutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = ...


TFAutoModelForMaskedLM = ...
class TFAutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = ...


TFAutoModelForSeq2SeqLM = ...
class TFAutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = ...


TFAutoModelForSequenceClassification = ...
class TFAutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = ...


TFAutoModelForQuestionAnswering = ...
class TFAutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = ...


TFAutoModelForTokenClassification = ...
class TFAutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = ...


TFAutoModelForMultipleChoice = ...
class TFAutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = ...


TFAutoModelForNextSentencePrediction = ...
class TFAutoModelWithLMHead(_TFAutoModelWithLMHead):
    @classmethod
    def from_config(cls, config):
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        ...
    


