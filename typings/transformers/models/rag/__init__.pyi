

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_tf_available, is_torch_available
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
from .tokenization_rag import RagTokenizer

_import_structure = ...
if is_torch_available():
    ...
if is_tf_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
