import os
from typing import Any, Dict, Optional, Union
from ...file_utils import is_sentencepiece_available, is_tokenizers_available
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ..albert.tokenization_albert import AlbertTokenizer
from ..albert.tokenization_albert_fast import AlbertTokenizerFast
from ..bart.tokenization_bart_fast import BartTokenizerFast
from ..barthez.tokenization_barthez import BarthezTokenizer
from ..barthez.tokenization_barthez_fast import BarthezTokenizerFast
from ..bert.tokenization_bert_fast import BertTokenizerFast
from ..bert_generation.tokenization_bert_generation import BertGenerationTokenizer
from ..big_bird.tokenization_big_bird import BigBirdTokenizer
from ..big_bird.tokenization_big_bird_fast import BigBirdTokenizerFast
from ..camembert.tokenization_camembert import CamembertTokenizer
from ..camembert.tokenization_camembert_fast import CamembertTokenizerFast
from ..convbert.tokenization_convbert_fast import ConvBertTokenizerFast
from ..cpm.tokenization_cpm import CpmTokenizer
from ..deberta.tokenization_deberta_fast import DebertaTokenizerFast
from ..deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer
from ..distilbert.tokenization_distilbert_fast import DistilBertTokenizerFast
from ..dpr.tokenization_dpr_fast import DPRQuestionEncoderTokenizerFast
from ..electra.tokenization_electra_fast import ElectraTokenizerFast
from ..funnel.tokenization_funnel_fast import FunnelTokenizerFast
from ..gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from ..herbert.tokenization_herbert_fast import HerbertTokenizerFast
from ..layoutlm.tokenization_layoutlm_fast import LayoutLMTokenizerFast
from ..led.tokenization_led_fast import LEDTokenizerFast
from ..longformer.tokenization_longformer_fast import LongformerTokenizerFast
from ..lxmert.tokenization_lxmert_fast import LxmertTokenizerFast
from ..m2m_100 import M2M100Tokenizer
from ..marian.tokenization_marian import MarianTokenizer
from ..mbart.tokenization_mbart import MBartTokenizer
from ..mbart.tokenization_mbart50 import MBart50Tokenizer
from ..mbart.tokenization_mbart50_fast import MBart50TokenizerFast
from ..mbart.tokenization_mbart_fast import MBartTokenizerFast
from ..mobilebert.tokenization_mobilebert_fast import MobileBertTokenizerFast
from ..mpnet.tokenization_mpnet_fast import MPNetTokenizerFast
from ..mt5 import MT5Tokenizer, MT5TokenizerFast
from ..openai.tokenization_openai_fast import OpenAIGPTTokenizerFast
from ..pegasus.tokenization_pegasus import PegasusTokenizer
from ..pegasus.tokenization_pegasus_fast import PegasusTokenizerFast
from ..reformer.tokenization_reformer import ReformerTokenizer
from ..reformer.tokenization_reformer_fast import ReformerTokenizerFast
from ..retribert.tokenization_retribert_fast import RetriBertTokenizerFast
from ..roberta.tokenization_roberta_fast import RobertaTokenizerFast
from ..roformer.tokenization_roformer_fast import RoFormerTokenizerFast
from ..speech_to_text import Speech2TextTokenizer
from ..squeezebert.tokenization_squeezebert_fast import SqueezeBertTokenizerFast
from ..t5.tokenization_t5 import T5Tokenizer
from ..t5.tokenization_t5_fast import T5TokenizerFast
from ..xlm_prophetnet.tokenization_xlm_prophetnet import XLMProphetNetTokenizer
from ..xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer
from ..xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast
from ..xlnet.tokenization_xlnet import XLNetTokenizer
from ..xlnet.tokenization_xlnet_fast import XLNetTokenizerFast
from .auto_factory import PathLike
from .configuration_auto import replace_list_option_in_docstrings

if is_sentencepiece_available(): ...
else:
    AlbertTokenizer = ...
    BarthezTokenizer = ...
    BertGenerationTokenizer = ...
    BigBirdTokenizer = ...
    CamembertTokenizer = ...
    CpmTokenizer = ...
    DebertaV2Tokenizer = ...
    MarianTokenizer = ...
    MBartTokenizer = ...
    MBart50Tokenizer = ...
    MT5Tokenizer = ...
    PegasusTokenizer = ...
    ReformerTokenizer = ...
    T5Tokenizer = ...
    XLMRobertaTokenizer = ...
    XLNetTokenizer = ...
    XLMProphetNetTokenizer = ...
    M2M100Tokenizer = ...
    Speech2TextTokenizer = ...
if is_tokenizers_available(): ...
else:
    AlbertTokenizerFast = ...
    BartTokenizerFast = ...
    BarthezTokenizerFast = ...
    BertTokenizerFast = ...
    BigBirdTokenizerFast = ...
    CamembertTokenizerFast = ...
    ConvBertTokenizerFast = ...
    DebertaTokenizerFast = ...
    DistilBertTokenizerFast = ...
    DPRQuestionEncoderTokenizerFast = ...
    ElectraTokenizerFast = ...
    FunnelTokenizerFast = ...
    GPT2TokenizerFast = ...
    HerbertTokenizerFast = ...
    LayoutLMTokenizerFast = ...
    LEDTokenizerFast = ...
    LongformerTokenizerFast = ...
    LxmertTokenizerFast = ...
    MBartTokenizerFast = ...
    MBart50TokenizerFast = ...
    MobileBertTokenizerFast = ...
    MPNetTokenizerFast = ...
    MT5TokenizerFast = ...
    OpenAIGPTTokenizerFast = ...
    PegasusTokenizerFast = ...
    ReformerTokenizerFast = ...
    RetriBertTokenizerFast = ...
    RobertaTokenizerFast = ...
    RoFormerTokenizerFast = ...
    SqueezeBertTokenizerFast = ...
    T5TokenizerFast = ...
    XLMRobertaTokenizerFast = ...
    XLNetTokenizerFast = ...
    PreTrainedTokenizerFast = ...
logger = ...
TOKENIZER_MAPPING = ...
NO_CONFIG_TOKENIZER = ...
SLOW_TOKENIZER_MAPPING = ...

def tokenizer_class_from_name(class_name: str): ...
def get_tokenizer_config(
    pretrained_model_name_or_path: PathLike,
    cache_dir: Optional[Union[str, os.PathLike]] = ...,
    force_download: bool = ...,
    resume_download: bool = ...,
    proxies: Optional[Dict[str, str]] = ...,
    use_auth_token: Optional[Union[bool, str]] = ...,
    revision: Optional[str] = ...,
    local_files_only: bool = ...,
    **kwargs
): ...

class AutoTokenizer:
    @classmethod
    @replace_list_option_in_docstrings(SLOW_TOKENIZER_MAPPING)
    def from_pretrained(
        cls, pretrained_model_name_or_path: PathLike, *inputs: Any, **kwargs: Any
    ) -> Union[PreTrainedTokenizerFast, PreTrainedTokenizer]: ...
