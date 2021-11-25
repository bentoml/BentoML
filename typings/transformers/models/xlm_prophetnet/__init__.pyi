

from ...file_utils import is_sentencepiece_available, is_torch_available
from .configuration_xlm_prophetnet import (
    XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLMProphetNetConfig,
)
from .modeling_xlm_prophetnet import (
    XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
    XLMProphetNetDecoder,
    XLMProphetNetEncoder,
    XLMProphetNetForCausalLM,
    XLMProphetNetForConditionalGeneration,
    XLMProphetNetModel,
)
from .tokenization_xlm_prophetnet import XLMProphetNetTokenizer

if is_sentencepiece_available():
    ...
if is_torch_available():
    ...
