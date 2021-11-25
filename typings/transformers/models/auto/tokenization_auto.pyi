

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

if is_sentencepiece_available():
    ...
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
if is_tokenizers_available():
    ...
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
def tokenizer_class_from_name(class_name: str):
    ...

def get_tokenizer_config(pretrained_model_name_or_path: PathLike, cache_dir: Optional[Union[str, os.PathLike]] = ..., force_download: bool = ..., resume_download: bool = ..., proxies: Optional[Dict[str, str]] = ..., use_auth_token: Optional[Union[bool, str]] = ..., revision: Optional[str] = ..., local_files_only: bool = ..., **kwargs):
    """
    Loads the tokenizer configuration from a pretrained model tokenizer configuration.

    Args:
        pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
            This can be either:

            - a string, the `model id` of a pretrained model configuration hosted inside a model repo on
              huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
              namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
            - a path to a `directory` containing a configuration file saved using the
              :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g., ``./my_model_directory/``.

        cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (:obj:`Dict[str, str]`, `optional`):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        use_auth_token (:obj:`str` or `bool`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
        revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
            identifier allowed by git.
        local_files_only (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, will only try to load the tokenizer configuration from local files.

    .. note::

        Passing :obj:`use_auth_token=True` is required when you want to use a private model.


    Returns:
        :obj:`Dict`: The configuration of the tokenizer.

    Examples::

        # Download configuration from huggingface.co and cache.
        tokenizer_config = get_tokenizer_config("bert-base-uncased")
        # This model does not have a tokenizer config so the result will be an empty dict.
        tokenizer_config = get_tokenizer_config("xlm-roberta-base")

        # Save a pretrained tokenizer locally and you can reload its config
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        tokenizer.save_pretrained("tokenizer-test")
        tokenizer_config = get_tokenizer_config("tokenizer-test")
    """
    ...

class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the :meth:`AutoTokenizer.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """
    @classmethod
    @replace_list_option_in_docstrings(SLOW_TOKENIZER_MAPPING)
    def from_pretrained(cls, pretrained_model_name_or_path: PathLike, *inputs: Any, **kwargs: Any) -> Union[PreTrainedTokenizerFast, PreTrainedTokenizer]:
        r"""
        Instantiate one of the tokenizer classes of the library from a pretrained model vocabulary.

        The tokenizer class to instantiate is selected based on the :obj:`model_type` property of the config object
        (either passed as an argument or loaded from :obj:`pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on :obj:`pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                Can be either:

                    - A string, the `model id` of a predefined tokenizer hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing vocabulary files required by the tokenizer, for instance saved
                      using the :func:`~transformers.PreTrainedTokenizer.save_pretrained` method, e.g.,
                      ``./my_model_directory/``.
                    - A path or url to a single saved vocabulary file if and only if the tokenizer only requires a
                      single vocabulary file (like Bert or XLNet), e.g.: ``./my_model_directory/vocab.txt``. (Not
                      applicable to all derived classes)
            inputs (additional positional arguments, `optional`):
                Will be passed along to the Tokenizer ``__init__()`` method.
            config (:class:`~transformers.PreTrainedConfig`, `optional`)
                The configuration object used to dertermine the tokenizer class to instantiate.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            subfolder (:obj:`str`, `optional`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
                facebook/rag-token-base), specify it here.
            use_fast (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to try to load the fast version of the tokenizer.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the Tokenizer ``__init__()`` method. Can be used to set special tokens like
                ``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``,
                ``mask_token``, ``additional_special_tokens``. See parameters in the ``__init__()`` for more details.

        Examples::

            >>> from transformers import AutoTokenizer

            >>> # Download vocabulary from huggingface.co and cache.
            >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

            >>> # Download vocabulary from huggingface.co (user-uploaded) and cache.
            >>> tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')

            >>> # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
            >>> tokenizer = AutoTokenizer.from_pretrained('./test/bert_saved_model/')

        """
        ...
    


