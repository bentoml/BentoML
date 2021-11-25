

from typing import List, Optional, Tuple

import numpy as np

from ...file_utils import is_datasets_available, is_faiss_available
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding

if is_datasets_available():
    ...
if is_faiss_available():
    ...
logger = ...
LEGACY_INDEX_PATH = ...
class Index:
    """
    A base class for the Indices encapsulated by the :class:`~transformers.RagRetriever`.
    """
    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        """
        Returns a list of dictionaries, containing titles and text of the retrieved documents.

        Args:
            doc_ids (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`):
                A tensor of document indices.
        """
        ...
    
    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=...) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each query in the batch, retrieves ``n_docs`` documents.

        Args:
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size):
                An array of query vectors.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Returns:
            :obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`: A tensor of indices of retrieved documents.
            :obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`: A tensor of vector representations of
            retrieved documents.
        """
        ...
    
    def is_initialized(self):
        """
        Returns :obj:`True` if index is already initialized.
        """
        ...
    
    def init_index(self):
        """
        A function responsible for loading the index into memory. Should be called only once per training run of a RAG
        model. E.g. if the model is trained on multiple GPUs in a distributed setup, only one of the workers will load
        the index.
        """
        ...
    


class LegacyIndex(Index):
    """
    An index which can be deserialized from the files built using https://github.com/facebookresearch/DPR. We use
    default faiss index parameters as specified in that repository.

    Args:
        vector_size (:obj:`int`):
            The dimension of indexed vectors.
        index_path (:obj:`str`):
            A path to a `directory` containing index files compatible with
            :class:`~transformers.models.rag.retrieval_rag.LegacyIndex`
    """
    INDEX_FILENAME = ...
    PASSAGE_FILENAME = ...
    def __init__(self, vector_size, index_path) -> None:
        ...
    
    def is_initialized(self):
        ...
    
    def init_index(self):
        ...
    
    def get_doc_dicts(self, doc_ids: np.array):
        ...
    
    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=...) -> Tuple[np.ndarray, np.ndarray]:
        ...
    


class HFIndexBase(Index):
    def __init__(self, vector_size, dataset, index_initialized=...) -> None:
        ...
    
    def init_index(self):
        ...
    
    def is_initialized(self):
        ...
    
    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        ...
    
    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=...) -> Tuple[np.ndarray, np.ndarray]:
        ...
    


class CanonicalHFIndex(HFIndexBase):
    """
    A wrapper around an instance of :class:`~datasets.Datasets`. If ``index_path`` is set to ``None``, we load the
    pre-computed index available with the :class:`~datasets.arrow_dataset.Dataset`, otherwise, we load the index from
    the indicated path on disk.

    Args:
        vector_size (:obj:`int`): the dimension of the passages embeddings used by the index
        dataset_name (:obj:`str`, optional, defaults to ``wiki_dpr``):
            A dataset identifier of the indexed dataset on HuggingFace AWS bucket (list all available datasets and ids
            with ``datasets.list_datasets()``).
        dataset_split (:obj:`str`, optional, defaults to ``train``)
            Which split of the ``dataset`` to load.
        index_name (:obj:`str`, optional, defaults to ``train``)
            The index_name of the index associated with the ``dataset``. The index loaded from ``index_path`` will be
            saved under this name.
        index_path (:obj:`str`, optional, defaults to ``None``)
            The path to the serialized faiss index on disk.
        use_dummy_dataset (:obj:`bool`, optional, defaults to ``False``): If True, use the dummy configuration of the dataset for tests.
    """
    def __init__(self, vector_size: int, dataset_name: str = ..., dataset_split: str = ..., index_name: Optional[str] = ..., index_path: Optional[str] = ..., use_dummy_dataset=...) -> None:
        ...
    
    def init_index(self):
        ...
    


class CustomHFIndex(HFIndexBase):
    """
    A wrapper around an instance of :class:`~datasets.Datasets`. The dataset and the index are both loaded from the
    indicated paths on disk.

    Args:
        vector_size (:obj:`int`): the dimension of the passages embeddings used by the index
        dataset_path (:obj:`str`):
            The path to the serialized dataset on disk. The dataset should have 3 columns: title (str), text (str) and
            embeddings (arrays of dimension vector_size)
        index_path (:obj:`str`)
            The path to the serialized faiss index on disk.
    """
    def __init__(self, vector_size: int, dataset, index_path=...) -> None:
        ...
    
    @classmethod
    def load_from_disk(cls, vector_size, dataset_path, index_path):
        ...
    
    def init_index(self):
        ...
    


class RagRetriever:
    """
    Retriever used to get documents from vector queries. It retrieves the documents embeddings as well as the documents
    contents, and it formats them to be used with a RagModel.

    Args:
        config (:class:`~transformers.RagConfig`):
            The configuration of the RAG model this Retriever is used with. Contains parameters indicating which
            ``Index`` to build. You can load your own custom dataset with ``config.index_name="custom"`` or use a
            canonical one (default) from the datasets library with ``config.index_name="wiki_dpr"`` for example.
        question_encoder_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer that was used to tokenize the question. It is used to decode the question and then use the
            generator_tokenizer.
        generator_tokenizer (:class:`~transformers.PreTrainedTokenizer`):
            The tokenizer used for the generator part of the RagModel.
        index (:class:`~transformers.models.rag.retrieval_rag.Index`, optional, defaults to the one defined by the configuration):
            If specified, use this index instead of the one built using the configuration

    Examples::

        >>> # To load the default "wiki_dpr" dataset with 21M passages from wikipedia (index name is 'compressed' or 'exact')
        >>> from transformers import RagRetriever
        >>> retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', dataset="wiki_dpr", index_name='compressed')

        >>> # To load your own indexed dataset built with the datasets library. More info on how to build the indexed dataset in examples/rag/use_own_knowledge_dataset.py
        >>> from transformers import RagRetriever
        >>> dataset = ...  # dataset must be a datasets.Datasets object with columns "title", "text" and "embeddings", and it must have a faiss index
        >>> retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', indexed_dataset=dataset)

        >>> # To load your own indexed dataset built with the datasets library that was saved on disk. More info in examples/rag/use_own_knowledge_dataset.py
        >>> from transformers import RagRetriever
        >>> dataset_path = "path/to/my/dataset"  # dataset saved via `dataset.save_to_disk(...)`
        >>> index_path = "path/to/my/index.faiss"  # faiss index saved via `dataset.get_index("embeddings").save(...)`
        >>> retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', index_name='custom', passages_path=dataset_path, index_path=index_path)

        >>> # To load the legacy index built originally for Rag's paper
        >>> from transformers import RagRetriever
        >>> retriever = RagRetriever.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', index_name='legacy')

    """
    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=..., init_retrieval=...) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, retriever_name_or_path, indexed_dataset=..., **kwargs):
        ...
    
    def save_pretrained(self, save_directory):
        ...
    
    def init_retrieval(self):
        """
        Retriever initialization function. It loads the index into memory.
        """
        ...
    
    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=...):
        r"""
        Postprocessing retrieved ``docs`` and combining them with ``input_strings``.

        Args:
            docs  (:obj:`dict`):
                Retrieved documents.
            input_strings (:obj:`str`):
                Input strings decoded by ``preprocess_query``.
            prefix (:obj:`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.

        Return:
            :obj:`tuple(tensors)`: a tuple consisting of two elements: contextualized ``input_ids`` and a compatible
            ``attention_mask``.
        """
        ...
    
    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> Tuple[np.ndarray, List[dict]]:
        """
        Retrieves documents for specified ``question_hidden_states``.

        Args:
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`):
                A batch of query vectors to retrieve with.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Return:
            :obj:`Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:

            - **retrieved_doc_embeds** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs, dim)`) -- The retrieval
              embeddings of the retrieved docs per query.
            - **doc_ids** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`) -- The ids of the documents in the
              index
            - **doc_dicts** (:obj:`List[dict]`): The :obj:`retrieved_doc_embeds` examples per query.
        """
        ...
    
    def set_ctx_encoder_tokenizer(self, ctx_encoder_tokenizer: PreTrainedTokenizer):
        ...
    
    def __call__(self, question_input_ids: List[List[int]], question_hidden_states: np.ndarray, prefix=..., n_docs=..., return_tensors=...) -> BatchEncoding:
        """
        Retrieves documents for specified :obj:`question_hidden_states`.

        Args:
            question_input_ids: (:obj:`List[List[int]]`) batch of input ids
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix: (:obj:`str`, `optional`):
                The prefix used by the generator's tokenizer.
            n_docs (:obj:`int`, `optional`):
                The number of docs retrieved per query.
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.

        Returns: :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following
        fields:

            - **context_input_ids** -- List of token ids to be fed to a model.

              `What are input IDs? <../glossary.html#input-ids>`__

            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model
            (when :obj:`return_attention_mask=True` or if `"attention_mask"` is in :obj:`self.model_input_names`).

              `What are attention masks? <../glossary.html#attention-mask>`__

            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        """
        ...
    


