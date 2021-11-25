


class PyTorchBenchmark:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class PyTorchBenchmarkArguments:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class DataCollator:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class DataCollatorForLanguageModeling:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DataCollatorForPermutationLanguageModeling:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DataCollatorForSeq2Seq:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class DataCollatorForSOP:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class DataCollatorForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DataCollatorForWholeWordMask:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class DataCollatorWithPadding:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


def default_data_collator(*args, **kwargs):
    ...

class GlueDataset:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class GlueDataTrainingArguments:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class LineByLineTextDataset:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class LineByLineWithRefDataset:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class LineByLineWithSOPTextDataset:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class SquadDataset:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class SquadDataTrainingArguments:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class TextDataset:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class TextDatasetForNextSentencePrediction:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class BeamScorer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class BeamSearchScorer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class ForcedBOSTokenLogitsProcessor:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ForcedEOSTokenLogitsProcessor:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class HammingDiversityLogitsProcessor:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class InfNanRemoveLogitsProcessor:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LogitsProcessor:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LogitsProcessorList:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LogitsWarper:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class MinLengthLogitsProcessor:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class NoBadWordsLogitsProcessor:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class NoRepeatNGramLogitsProcessor:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class PrefixConstrainedLogitsProcessor:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RepetitionPenaltyLogitsProcessor:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class TemperatureLogitsWarper:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class TopKLogitsWarper:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class TopPLogitsWarper:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class MaxLengthCriteria:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class MaxTimeCriteria:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class StoppingCriteria:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class StoppingCriteriaList:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


def top_k_top_p_filtering(*args, **kwargs):
    ...

class Conv1D:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class PreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def apply_chunking_to_forward(*args, **kwargs):
    ...

def prune_layer(*args, **kwargs):
    ...

ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class AlbertForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AlbertForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AlbertForPreTraining:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class AlbertForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AlbertForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AlbertForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AlbertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AlbertPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_albert(*args, **kwargs):
    ...

MODEL_FOR_CAUSAL_LM_MAPPING = ...
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = ...
MODEL_FOR_MASKED_LM_MAPPING = ...
MODEL_FOR_MULTIPLE_CHOICE_MAPPING = ...
MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = ...
MODEL_FOR_OBJECT_DETECTION_MAPPING = ...
MODEL_FOR_PRETRAINING_MAPPING = ...
MODEL_FOR_QUESTION_ANSWERING_MAPPING = ...
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = ...
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = ...
MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = ...
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = ...
MODEL_MAPPING = ...
MODEL_WITH_LM_HEAD_MAPPING = ...
class AutoModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AutoModelForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AutoModelForImageClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AutoModelForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AutoModelForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AutoModelForNextSentencePrediction:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AutoModelForPreTraining:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AutoModelForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AutoModelForSeq2SeqLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AutoModelForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AutoModelForTableQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AutoModelForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class AutoModelWithLMHead:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


BART_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class BartForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BartForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BartForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BartForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BartModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BartPretrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class PretrainedBartModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


BERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class BertForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BertForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BertForNextSentencePrediction:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class BertForPreTraining:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class BertForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BertForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BertForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BertLayer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class BertLMHeadModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BertPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_bert(*args, **kwargs):
    ...

class BertGenerationDecoder:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class BertGenerationEncoder:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class BertGenerationPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_bert_generation(*args, **kwargs):
    ...

BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class BigBirdForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BigBirdForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BigBirdForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BigBirdForPreTraining:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class BigBirdForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BigBirdForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BigBirdForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BigBirdLayer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class BigBirdModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BigBirdPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_big_bird(*args, **kwargs):
    ...

BIGBIRD_PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class BigBirdPegasusForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BigBirdPegasusForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BigBirdPegasusForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BigBirdPegasusForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BigBirdPegasusModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BigBirdPegasusPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class BlenderbotForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BlenderbotForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BlenderbotModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BlenderbotPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class BlenderbotSmallForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BlenderbotSmallForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BlenderbotSmallModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class BlenderbotSmallPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class CamembertForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CamembertForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CamembertForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CamembertForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CamembertForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CamembertForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CamembertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


CANINE_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class CanineForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CanineForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CanineForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CanineForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CanineLayer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class CanineModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CaninePreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_canine(*args, **kwargs):
    ...

CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class CLIPModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CLIPPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CLIPTextModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CLIPVisionModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class ConvBertForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ConvBertForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ConvBertForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ConvBertForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ConvBertForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ConvBertLayer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class ConvBertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ConvBertPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_convbert(*args, **kwargs):
    ...

CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class CTRLForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CTRLLMHeadModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CTRLModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class CTRLPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class DebertaForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DebertaForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DebertaForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DebertaForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DebertaModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DebertaPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class DebertaV2ForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DebertaV2ForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DebertaV2ForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DebertaV2ForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DebertaV2Model:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DebertaV2PreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


DEIT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class DeiTForImageClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class DeiTForImageClassificationWithTeacher:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class DeiTModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DeiTPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class DistilBertForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DistilBertForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DistilBertForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DistilBertForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DistilBertForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DistilBertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class DistilBertPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class DPRContextEncoder:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class DPRPretrainedContextEncoder:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class DPRPretrainedQuestionEncoder:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class DPRPretrainedReader:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class DPRQuestionEncoder:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class DPRReader:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class ElectraForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ElectraForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ElectraForPreTraining:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class ElectraForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ElectraForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ElectraForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ElectraModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ElectraPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_electra(*args, **kwargs):
    ...

class EncoderDecoderModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class FlaubertForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FlaubertForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FlaubertForQuestionAnsweringSimple:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FlaubertForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FlaubertForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FlaubertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FlaubertWithLMHeadModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FSMTForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FSMTModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class PretrainedFSMTModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class FunnelBaseModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FunnelForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FunnelForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FunnelForPreTraining:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class FunnelForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FunnelForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FunnelForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FunnelModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class FunnelPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_funnel(*args, **kwargs):
    ...

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class GPT2DoubleHeadsModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class GPT2ForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class GPT2LMHeadModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class GPT2Model:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class GPT2PreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_gpt2(*args, **kwargs):
    ...

GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class GPTNeoForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class GPTNeoForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class GPTNeoModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class GPTNeoPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_gpt_neo(*args, **kwargs):
    ...

HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class HubertForCTC:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class HubertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class HubertPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


IBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class IBertForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class IBertForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class IBertForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class IBertForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class IBertForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class IBertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class IBertPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class LayoutLMForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LayoutLMForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LayoutLMForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LayoutLMModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LayoutLMPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


LED_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class LEDForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LEDForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LEDForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LEDModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LEDPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class LongformerForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LongformerForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LongformerForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LongformerForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LongformerForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LongformerModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LongformerPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LongformerSelfAttention:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


LUKE_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class LukeForEntityClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class LukeForEntityPairClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class LukeForEntitySpanClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class LukeModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LukePreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LxmertEncoder:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class LxmertForPreTraining:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class LxmertForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LxmertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LxmertPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class LxmertVisualFeatureEncoder:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class LxmertXLayer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class M2M100ForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class M2M100Model:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class M2M100PreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MarianForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MarianModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MarianMTModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MBartForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MBartForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MBartForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MBartForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MBartModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MBartPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


MEGATRON_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class MegatronBertForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MegatronBertForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MegatronBertForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MegatronBertForNextSentencePrediction:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class MegatronBertForPreTraining:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class MegatronBertForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MegatronBertForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MegatronBertForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MegatronBertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MegatronBertPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MMBTForClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class MMBTModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ModalEmbeddings:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class MobileBertForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MobileBertForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MobileBertForNextSentencePrediction:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class MobileBertForPreTraining:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class MobileBertForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MobileBertForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MobileBertForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MobileBertLayer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class MobileBertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MobileBertPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_mobilebert(*args, **kwargs):
    ...

MPNET_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class MPNetForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MPNetForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MPNetForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MPNetForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MPNetForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MPNetLayer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class MPNetModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MPNetPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MT5EncoderModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MT5ForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class MT5Model:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class OpenAIGPTDoubleHeadsModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class OpenAIGPTForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class OpenAIGPTLMHeadModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class OpenAIGPTModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class OpenAIGPTPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_openai_gpt(*args, **kwargs):
    ...

class PegasusForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class PegasusForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class PegasusModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class PegasusPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class ProphetNetDecoder:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class ProphetNetEncoder:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class ProphetNetForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ProphetNetForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ProphetNetModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ProphetNetPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RagModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RagPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RagSequenceForGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class RagTokenForGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class ReformerAttention:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class ReformerForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ReformerForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ReformerForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ReformerLayer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class ReformerModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ReformerModelWithLMHead:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ReformerPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class RetriBertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RetriBertPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class RobertaForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RobertaForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RobertaForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RobertaForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RobertaForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RobertaForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RobertaModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RobertaPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class RoFormerForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RoFormerForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RoFormerForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RoFormerForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RoFormerForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RoFormerForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RoFormerLayer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class RoFormerModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class RoFormerPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_roformer(*args, **kwargs):
    ...

SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class Speech2TextForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class Speech2TextModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class Speech2TextPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class SqueezeBertForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class SqueezeBertForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class SqueezeBertForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class SqueezeBertForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class SqueezeBertForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class SqueezeBertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class SqueezeBertModule:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class SqueezeBertPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


T5_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class T5EncoderModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class T5ForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class T5Model:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class T5PreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_t5(*args, **kwargs):
    ...

TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class TapasForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class TapasForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class TapasForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class TapasModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class TapasPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class AdaptiveEmbedding:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class TransfoXLForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class TransfoXLLMHeadModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class TransfoXLModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class TransfoXLPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_transfo_xl(*args, **kwargs):
    ...

VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class VisualBertForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class VisualBertForPreTraining:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class VisualBertForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class VisualBertForRegionToPhraseAlignment:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class VisualBertForVisualReasoning:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class VisualBertLayer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class VisualBertModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class VisualBertPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


VIT_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class ViTForImageClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class ViTModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class ViTPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class Wav2Vec2ForCTC:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class Wav2Vec2ForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class Wav2Vec2ForPreTraining:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class Wav2Vec2Model:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class Wav2Vec2PreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


XLM_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class XLMForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMForQuestionAnsweringSimple:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMWithLMHeadModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class XLMProphetNetDecoder:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class XLMProphetNetEncoder:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class XLMProphetNetForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMProphetNetForConditionalGeneration:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMProphetNetModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class XLMRobertaForCausalLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMRobertaForMaskedLM:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMRobertaForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMRobertaForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMRobertaForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMRobertaForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLMRobertaModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = ...
class XLNetForMultipleChoice:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLNetForQuestionAnswering:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLNetForQuestionAnsweringSimple:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLNetForSequenceClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLNetForTokenClassification:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLNetLMHeadModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLNetModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


class XLNetPreTrainedModel:
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        ...
    


def load_tf_weights_in_xlnet(*args, **kwargs):
    ...

class Adafactor:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


class AdamW:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


def get_constant_schedule(*args, **kwargs):
    ...

def get_constant_schedule_with_warmup(*args, **kwargs):
    ...

def get_cosine_schedule_with_warmup(*args, **kwargs):
    ...

def get_cosine_with_hard_restarts_schedule_with_warmup(*args, **kwargs):
    ...

def get_linear_schedule_with_warmup(*args, **kwargs):
    ...

def get_polynomial_decay_schedule_with_warmup(*args, **kwargs):
    ...

def get_scheduler(*args, **kwargs):
    ...

class Trainer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


def torch_distributed_zero_first(*args, **kwargs):
    ...

class Seq2SeqTrainer:
    def __init__(self, *args, **kwargs) -> None:
        ...
    


