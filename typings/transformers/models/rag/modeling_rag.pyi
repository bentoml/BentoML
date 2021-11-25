

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from ...configuration_utils import PretrainedConfig
from ...file_utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever

"""RAG model implementation."""
logger = ...
_CONFIG_FOR_DOC = ...
@dataclass
class RetrievAugLMMarginOutput(ModelOutput):
    """
    Base class for retriever augmented marginalized models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
            :obj:`question_encoder_last_hidden_state`.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see :obj:`past_key_values` input) to speed up sequential decoding.
        retrieved_doc_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs, hidden_size)`, `optional`, returned when `output_retrieved=True`):
            Embedded documents retrieved by the retriever. Is used with ``question_encoder_last_hidden_state`` to
            compute the ``doc_scores``.
        retrieved_doc_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, config.n_docs)`, `optional`, returned when `output_retrieved=True`):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by the
            retriever.
        question_encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """
    loss: Optional[torch.FloatTensor] = ...
    logits: torch.FloatTensor = ...
    doc_scores: torch.FloatTensor = ...
    past_key_values: Optional[List[torch.FloatTensor]] = ...
    retrieved_doc_embeds: Optional[torch.FloatTensor] = ...
    retrieved_doc_ids: Optional[torch.LongTensor] = ...
    context_input_ids: Optional[torch.LongTensor] = ...
    context_attention_mask: Optional[torch.LongTensor] = ...
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = ...
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    question_enc_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = ...
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor]] = ...


@dataclass
class RetrievAugLMOutput(ModelOutput):
    """
    Args:
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
            :obj:`question_encoder_last_hidden_state`.
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2,
            batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see :obj:`past_key_values` input) to speed up sequential decoding.
        retrieved_doc_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs, hidden_size)`, `optional`, returned when `output_retrieved=True`):
            Embedded documents retrieved by the retriever. Is used with ``question_encoder_last_hidden_state`` to
            compute the ``doc_scores``.
        retrieved_doc_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, config.n_docs)`, `optional`, returned when `output_retrieved=True`):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by the
            retriever.
        question_encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings and one for the output of each
            layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Cross-attentions weights of the generator decoder, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """
    logits: torch.FloatTensor = ...
    doc_scores: torch.FloatTensor = ...
    past_key_values: Optional[List[torch.FloatTensor]] = ...
    retrieved_doc_embeds: Optional[torch.FloatTensor] = ...
    retrieved_doc_ids: Optional[torch.LongTensor] = ...
    context_input_ids: Optional[torch.LongTensor] = ...
    context_attention_mask: Optional[torch.LongTensor] = ...
    question_encoder_last_hidden_state: Optional[torch.FloatTensor] = ...
    question_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    question_enc_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    generator_enc_last_hidden_state: Optional[torch.FloatTensor] = ...
    generator_enc_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    generator_enc_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    generator_dec_hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    generator_dec_attentions: Optional[Tuple[torch.FloatTensor]] = ...
    generator_cross_attentions: Optional[Tuple[torch.FloatTensor]] = ...


class RagPreTrainedModel(PreTrainedModel):
    r"""
    RAG models were released with the paper `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    <https://arxiv.org/abs/2005.11401>`_ by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.

    RAG is a retriever augmented model and encapsulate three components: a question encoder, a dataset retriever and a
    generator, the encoder and generator are trainable while the retriever is just an indexed dataset.

    """
    config_class = RagConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    @classmethod
    def from_pretrained(cls, *args, **kwargs): # -> tuple[Unknown | RagPreTrainedModel, dict[str, Unbound | list[str] | list[Unknown]]] | RagPreTrainedModel:
        ...
    
    @classmethod
    def from_pretrained_question_encoder_generator(cls, question_encoder_pretrained_model_name_or_path: str = ..., generator_pretrained_model_name_or_path: str = ..., retriever: RagRetriever = ..., **kwargs) -> PreTrainedModel:
        r"""
        Instantiates an question encoder and a generator from one or two base classes of the library from pretrained
        model checkpoints.

        The model is set in evaluation mode by default using :obj:`model.eval()` (Dropout modules are deactivated). To
        train the model, you need to first set it back in training mode with :obj:`model.train()`.

        Params:
            question_encoder_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                Information necessary to initiate the question encoder. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            generator_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                Information necessary to initiate the generator. Can be either:

                    - A string, the `model id` of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like ``bert-base-uncased``, or namespaced under
                      a user or organization name, like ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
                      this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
                      as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in
                      a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
            retriever (:class:`~transformers.RagRetriever`, `optional`):
                The retriever to use.
            kwwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                ``output_attentions=True``).

                - To update the question_encoder configuration, use the prefix `question_encoder_` for each
                  configuration parameter.
                - To update the generator configuration, use the prefix `generator_` for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a :obj:`config` is provided or automatically loaded.

        Example::

            >>> from transformers import RagModel
            >>> # initialize a RAG from two pretrained models.
            >>> model = RagModel.from_question_encoder_generator_pretrained('facebook/dpr-question_encoder-single-nq-base', 't5-small')
            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./rag")
            >>> # load fine-tuned model
            >>> model = RagModel.from_pretrained("./rag")

        """
        ...
    


RAG_START_DOCSTRING = ...
RAG_FORWARD_INPUTS_DOCSTRING = ...
@add_start_docstrings_to_model_forward(RAG_START_DOCSTRING)
class RagModel(RagPreTrainedModel):
    def __init__(self, config: Optional[PretrainedConfig] = ..., question_encoder: Optional[PreTrainedModel] = ..., generator: Optional[PreTrainedModel] = ..., retriever: Optional = ..., **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., encoder_outputs=..., decoder_input_ids=..., decoder_attention_mask=..., past_key_values=..., doc_scores=..., context_input_ids=..., context_attention_mask=..., use_cache=..., output_attentions=..., output_hidden_states=..., output_retrieved=..., n_docs=...):
        r"""
        Returns:

        Example::

            >>> from transformers import RagTokenizer, RagRetriever, RagModel
            >>> import torch

            >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
            >>> retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)
            >>> # initialize with RagRetriever to do everything in one forward call
            >>> model = RagModel.from_pretrained("facebook/rag-token-base", retriever=retriever)

            >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
            >>> outputs = model(input_ids=inputs["input_ids"])
        """
        ...
    


@add_start_docstrings_to_model_forward("""
    A RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.
    """, RAG_START_DOCSTRING)
class RagSequenceForGeneration(RagPreTrainedModel):
    def __init__(self, config: Optional[PretrainedConfig] = ..., question_encoder: Optional[PreTrainedModel] = ..., generator: Optional[PreTrainedModel] = ..., retriever: Optional = ..., **kwargs) -> None:
        ...
    
    def set_retriever(self, retriever: RagRetriever): # -> None:
        ...
    
    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., encoder_outputs=..., decoder_input_ids=..., decoder_attention_mask=..., past_key_values=..., context_input_ids=..., context_attention_mask=..., doc_scores=..., use_cache=..., output_attentions=..., output_hidden_states=..., output_retrieved=..., exclude_bos_score=..., reduce_loss=..., labels=..., n_docs=..., **kwargs): # -> RetrievAugLMMarginOutput:
        r"""
        exclude_bos_score (:obj:`bool`, `optional`):
            Only relevant if ``labels`` is passed. If :obj:`True`, the score of the BOS token is disregarded when
            computing the loss.
        reduce_loss (:obj:`bool`, `optional`):
            Only relevant if ``labels`` is passed. If :obj:`True`, the NLL loss is reduced using the
            ``torch.Tensor.sum`` operation.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
             Legacy dictionary, which is required so that model can use `generate()` function.

        Returns:

        Example::

            >>> from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
            >>> import torch

            >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
            >>> retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
            >>> # initialize with RagRetriever to do everything in one forward call
            >>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

            >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
            >>> with tokenizer.as_target_tokenizer():
            ...    targets = tokenizer("In Paris, there are 10 million people.", return_tensors="pt")
            >>> input_ids = inputs["input_ids"]
            >>> labels = targets["input_ids"]
            >>> outputs = model(input_ids=input_ids, labels=labels)

            >>> # or use retriever separately
            >>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
            >>> # 1. Encode
            >>> question_hidden_states = model.question_encoder(input_ids)[0]
            >>> # 2. Retrieve
            >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
            >>> doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)
            >>> # 3. Forward to generator
            >>> outputs = model(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores, decoder_input_ids=labels)
        """
        ...
    
    @property
    def retriever(self): # -> RagRetriever:
        ...
    
    @property
    def generator(self): # -> PreTrainedModel | None:
        ...
    
    @property
    def question_encoder(self): # -> PreTrainedModel | None:
        ...
    
    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.LongTensor] = ..., context_input_ids=..., context_attention_mask=..., doc_scores=..., do_deduplication=..., num_return_sequences=..., num_beams=..., n_docs=..., **model_kwargs):
        """
        Implements RAG sequence "thorough" decoding. Read the
        :meth:`~transformers.generation_utils.GenerationMixin.generate`` documentation for more information on how to
        set other generate input parameters.

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`input_ids` is not passed, then
                :obj:`context_input_ids` has to be provided.
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            context_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Input IDs post-processed from the retrieved documents and the question encoder input_ids by the
                retriever.
            context_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by
                the retriever.

                If the model is not initialized with a ``retriever`` or ``input_ids`` is not given,
                :obj:`context_input_ids` and :obj:`context_attention_mask` have to be provided to the forward pass.
                They are returned by :meth:`~transformers.RagRetriever.__call__`.
            doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
                :obj:`question_encoder_last_hidden_state`.

                If the model is not initialized with a ``retriever`` or ``input_ids`` is not given, :obj:`doc_scores`
                has to be provided to the forward pass. :obj:`doc_scores` are returned by
                :meth:`~transformers.RagRetriever.__call__`.
            do_deduplication (:obj:`bool`, `optional`):
                Whether or not to deduplicate the generations from different context documents for a given input. Has
                to be set to :obj:`False` if used while training with distributed backend.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the ``generator``'s
                `:func:`~transformers.generation_utils.GenerationMixin.generate`` function, where we set
                ``num_return_sequences`` to :obj:`num_beams`.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            n_docs (:obj:`int`, `optional`, defaults to :obj:`config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            kwargs:
                Additional kwargs will be passed to :meth:`~transformers.generation_utils.GenerationMixin.generate`.

        Return:
            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence length) is either equal to :obj:`max_length` or shorter if all
            batches finished early due to the :obj:`eos_token_id`.
        """
        ...
    
    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=..., epsilon=..., exclude_bos_score=..., n_docs=...):
        ...
    


@add_start_docstrings_to_model_forward("""
    A RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.
    """, RAG_START_DOCSTRING)
class RagTokenForGeneration(RagPreTrainedModel):
    def __init__(self, config: Optional[PretrainedConfig] = ..., question_encoder: Optional[PreTrainedModel] = ..., generator: Optional[PreTrainedModel] = ..., retriever: Optional = ..., **kwargs) -> None:
        ...
    
    def set_retriever(self, retriever: RagRetriever): # -> None:
        ...
    
    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel): # -> None:
        ...
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past=..., attention_mask=..., use_cache=..., encoder_outputs=..., doc_scores=..., n_docs=..., **kwargs): # -> dict[str, Unknown | bool | None]:
        ...
    
    @property
    def retriever(self): # -> RagRetriever:
        ...
    
    @property
    def generator(self): # -> PreTrainedModel | None:
        ...
    
    @property
    def question_encoder(self): # -> PreTrainedModel | None:
        ...
    
    def marginalize(self, seq_logits, doc_scores, n_docs=...): # -> Tensor:
        ...
    
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., encoder_outputs=..., decoder_input_ids=..., decoder_attention_mask=..., past_key_values=..., context_input_ids=..., context_attention_mask=..., doc_scores=..., use_cache=..., output_attentions=..., output_hidden_states=..., output_retrieved=..., do_marginalize=..., reduce_loss=..., labels=..., n_docs=..., **kwargs): # -> RetrievAugLMMarginOutput:
        r"""
        do_marginalize (:obj:`bool`, `optional`):
            If :obj:`True`, the logits are marginalized over all documents by making use of
            ``torch.nn.functional.log_softmax``.
        reduce_loss (:obj:`bool`, `optional`):
            Only relevant if ``labels`` is passed. If :obj:`True`, the NLL loss is reduced using the
            ``torch.Tensor.sum`` operation.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Legacy dictionary, which is required so that model can use `generate()` function.

        Returns:

        Example::

            >>> from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
            >>> import torch

            >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
            >>> retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
            >>> # initialize with RagRetriever to do everything in one forward call
            >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

            >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
            >>> with tokenizer.as_target_tokenizer():
            ...    targets = tokenizer("In Paris, there are 10 million people.", return_tensors="pt")
            >>> input_ids = inputs["input_ids"]
            >>> labels = targets["input_ids"]
            >>> outputs = model(input_ids=input_ids, labels=labels)

            >>> # or use retriever separately
            >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
            >>> # 1. Encode
            >>> question_hidden_states = model.question_encoder(input_ids)[0]
            >>> # 2. Retrieve
            >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
            >>> doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)
            >>> # 3. Forward to generator
            >>> outputs = model(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores, decoder_input_ids=labels)

            >>> # or directly generate
            >>> generated = model.generate(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores)
            >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        """
        ...
    
    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.LongTensor] = ..., context_input_ids=..., context_attention_mask=..., doc_scores=..., max_length=..., min_length=..., early_stopping=..., use_cache=..., num_beams=..., num_beam_groups=..., diversity_penalty=..., bos_token_id=..., pad_token_id=..., eos_token_id=..., length_penalty=..., no_repeat_ngram_size=..., encoder_no_repeat_ngram_size=..., repetition_penalty=..., bad_words_ids=..., num_return_sequences=..., decoder_start_token_id=..., n_docs=..., prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = ..., forced_bos_token_id: Optional[int] = ..., forced_eos_token_id: Optional[int] = ..., remove_invalid_values: Optional[bool] = ..., **model_kwargs): # -> GreedySearchOutput | BeamSearchOutput | LongTensor:
        """
        Implements RAG token decoding.

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`input_ids` is not passed, then
                :obj:`context_input_ids` has to be provided.
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            context_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Input IDs post-processed from the retrieved documents and the question encoder :obj:`input_ids` by the
                retriever.

                If the model has is not initialized with a ``retriever``, :obj:`context_input_ids` has to be provided
                to the forward pass. :obj:`context_input_ids` are returned by
                :meth:`~transformers.RagRetriever.__call__`.
            context_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by
                the retriever.

                If the model has is not initialized with a ``retriever``, :obj:`context_input_ids` has to be provided
                to the forward pass. :obj:`context_input_ids` are returned by
                :meth:`~transformers.RagRetriever.__call__`.
            doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
                :obj:`question_encoder_last_hidden_state`.

                If the model has is not initialized with a ``retriever``, :obj:`context_input_ids` has to be provided
                to the forward pass. :obj:`context_input_ids` are returned by
                :meth:`~transformers.RagRetriever.__call__`.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to stop the beam search when at least ``num_beams`` sentences are finished per batch or
                not.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.

                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the ``encoder_input_ids`` cannot occur in the
                ``decoder_input_ids``.
            bad_words_ids(:obj:`List[int]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            num_beam_groups (:obj:`int`, `optional`, defaults to 1):
                Number of groups to divide :obj:`num_beams` into in order to ensure diversity among different groups of
                beams. `this paper <https://arxiv.org/pdf/1610.02424.pdf>`__ for more details.
            diversity_penalty (:obj:`float`, `optional`, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that :obj:`diversity_penalty` is only effective if ``group beam search`` is
                enabled.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the ``generator``'s
                `:func:`~transformers.generation_utils.GenerationMixin.generate` function, where we set
                ``num_return_sequences`` to :obj:`num_beams`.
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            n_docs (:obj:`int`, `optional`, defaults to :obj:`config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments :obj:`inputs_ids` and the batch ID
                :obj:`batch_id`. It has to return a list with the allowed tokens for the next generation step
                conditioned on the previously generated tokens :obj:`inputs_ids` and the batch ID :obj:`batch_id`. This
                argument is useful for constrained generation conditioned on the prefix, as described in
                `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.
            forced_bos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the first generated token after the :obj:`decoder_start_token_id`.
                Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token
                needs to be the target language token.
            forced_eos_token_id (:obj:`int`, `optional`):
                The id of the token to force as the last generated token when :obj:`max_length` is reached.
            remove_invalid_values (:obj:`bool`, `optional`):
                Whether to remove possible `nan` and `inf` outputs of the model to prevent the generation method to
                crash. Note that using ``remove_invalid_values`` can slow down generation.

        Return:
            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
            batches finished early due to the :obj:`eos_token_id`.
        """
        ...
    
    def get_input_embeddings(self): # -> Module:
        ...
    
    def get_output_embeddings(self): # -> Module:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> Any:
        ...
    
    def shift_tokens_right(self, input_ids, start_token_id=...):
        """Shift input ids one token to the right, and pad with start_token_id"""
        ...
    
    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=..., epsilon=..., n_docs=...):
        ...
    


