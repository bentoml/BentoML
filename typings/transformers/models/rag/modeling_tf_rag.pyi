

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from ...configuration_utils import PretrainedConfig
from ...file_utils import (
    ModelOutput,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_tf_utils import TFCausalLanguageModelingLoss, TFPreTrainedModel
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever

"""TFRAG model implementation."""
logger = ...
_CONFIG_FOR_DOC = ...
@dataclass
class TFRetrievAugLMMarginOutput(ModelOutput):
    """
    Base class for retriever augmented marginalized models outputs.

    Args:
        loss (:obj:`tf.Tensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        past_key_values (:obj:`List[tf.Tensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`tf.Tensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see :obj:`past_key_values` input) to speed up sequential decoding.
        doc_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
            :obj:`question_encoder_last_hidden_state`.
        retrieved_doc_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, config.n_docs, hidden_size)`, `optional`, returned when `output_retrieved=True`):
            Embedded documents retrieved by the retriever. Is used with ``question_encoder_last_hidden_state`` to
            compute the ``doc_scores``.
        retrieved_doc_ids (:obj:`tf.Tensor` (int32) of shape :obj:`(batch_size, config.n_docs)`, `optional`, returned when `output_retrieved=True`):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (:obj:`tf.Tensor`(int32) of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (:obj:`tf.Tensor` (int32) of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by the
            retriever.
        question_encoder_last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
    """
    loss: Optional[tf.Tensor] = ...
    logits: tf.Tensor = ...
    past_key_values: Optional[List[tf.Tensor]] = ...
    doc_scores: Optional[tf.Tensor] = ...
    retrieved_doc_embeds: Optional[tf.Tensor] = ...
    retrieved_doc_ids: Optional[tf.Tensor] = ...
    context_input_ids: Optional[tf.Tensor] = ...
    context_attention_mask: Optional[tf.Tensor] = ...
    question_encoder_last_hidden_state: Optional[tf.Tensor] = ...
    question_enc_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    question_enc_attentions: Optional[Tuple[tf.Tensor]] = ...
    generator_enc_last_hidden_state: Optional[tf.Tensor] = ...
    generator_enc_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    generator_enc_attentions: Optional[Tuple[tf.Tensor]] = ...
    generator_dec_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    generator_dec_attentions: Optional[Tuple[tf.Tensor]] = ...


@dataclass
class TFRetrievAugLMOutput(ModelOutput):
    """
    Args:
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head. The score is possibly marginalized over all documents for
            each vocabulary token.
        past_key_values (:obj:`List[tf.Tensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`tf.Tensor` of length :obj:`config.n_layers`, with each tensor of shape :obj:`(2, batch_size,
            num_heads, sequence_length, embed_size_per_head)`).

            Contains precomputed hidden-states (key and values in the attention blocks) of the decoder that can be used
            (see :obj:`past_key_values` input) to speed up sequential decoding.
        doc_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size, config.n_docs)`):
            Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
            :obj:`question_encoder_last_hidden_state`.
        retrieved_doc_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, config.n_docs, hidden_size)`, `optional`, returned when `output_retrieved=True`):
            Embedded documents retrieved by the retriever. Is used with ``question_encoder_last_hidden_state`` to
            compute the ``doc_scores``.
        retrieved_doc_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, config.n_docs)`, `optional`, returned when `output_retrieved=True`):
            The indexes of the embedded documents retrieved by the retriever.
        context_input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Input ids post-processed from the retrieved documents and the question encoder input_ids by the retriever.
        context_attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
            Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by the
            retriever.
        question_encoder_last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden states at the output of the last layer of the question encoder pooled output of the
            model.
        question_enc_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the question encoder at the output of each layer plus the initial embedding outputs.
        question_enc_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the question encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_enc_last_hidden_state (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the generator encoder of the model.
        generator_enc_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator encoder at the output of each layer plus the initial embedding outputs.
        generator_enc_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator encoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        generator_dec_hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden states of the generator decoder at the output of each layer plus the initial embedding outputs.
        generator_dec_attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the generator decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
    """
    logits: tf.Tensor = ...
    past_key_values: Optional[List[tf.Tensor]] = ...
    doc_scores: Optional[tf.Tensor] = ...
    retrieved_doc_embeds: Optional[tf.Tensor] = ...
    retrieved_doc_ids: Optional[tf.Tensor] = ...
    context_input_ids: Optional[tf.Tensor] = ...
    context_attention_mask: Optional[tf.Tensor] = ...
    question_encoder_last_hidden_state: Optional[tf.Tensor] = ...
    question_enc_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    question_enc_attentions: Optional[Tuple[tf.Tensor]] = ...
    generator_enc_last_hidden_state: Optional[tf.Tensor] = ...
    generator_enc_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    generator_enc_attentions: Optional[Tuple[tf.Tensor]] = ...
    generator_dec_hidden_states: Optional[Tuple[tf.Tensor]] = ...
    generator_dec_attentions: Optional[Tuple[tf.Tensor]] = ...


class TFRagPreTrainedModel(TFPreTrainedModel):
    r"""
    RAG models were released with the paper `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
    <https://arxiv.org/abs/2005.11401>`__ by Patrick Lewis, Ethan Perez, Aleksandra Piktus et al.

    RAG is a retriever augmented model and encapsulate three components: a question encoder, a dataset retriever and a
    generator, the encoder and generator are trainable while the retriever is just an indexed dataset.

    """
    config_class = RagConfig
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    @classmethod
    def from_pretrained_question_encoder_generator(cls, question_encoder_pretrained_model_name_or_path: str = ..., generator_pretrained_model_name_or_path: str = ..., retriever: RagRetriever = ..., *model_args, **kwargs) -> TFPreTrainedModel:
        r"""
        Instantiates an question encoder and a generator from one or two base classes of the library from pretrained
        model checkpoints.

        Params:
            question_encoder_pretrained_model_name_or_path (:obj: `str`, `optional`):
                Information necessary to initiate the question encoder. Can be either:

                    - A string with the `shortcut name` of a pretrained model to load from cache or download, e.g.,
                      ``bert-base-uncased``.
                    - A string with the `identifier name` of a pretrained model that was user-uploaded to our S3, e.g.,
                      ``dbmdz/bert-base-german-cased``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `pytorch index checkpoint file` (e.g, ``./pt_model/``). In this case,
                      ``question_encoder_from_pt`` should be set to :obj:`True`.

            generator_pretrained_model_name_or_path (:obj: `str`, `optional`, defaults to `None`):
                Information necessary to initiate the generator. Can be either:

                    - A string with the `shortcut name` of a pretrained model to load from cache or download, e.g.,
                      ``t5-small``.
                    - A string with the `identifier name` of a pretrained model that was user-uploaded to our S3, e.g.,
                      ``facebook/bart-base``.
                    - A path to a `directory` containing model weights saved using
                      :func:`~transformers.TFPreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
                    - A path or url to a `pytorch checkpoint file` (e.g, ``./pt_model/``). In this case,
                      ``generator_from_pt`` should be set to :obj:`True`.

            model_args (remaining positional arguments, `optional`):
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
            retriever (:class:`~transformers.RagRetriever`, `optional`):
                The retriever to use.
            kwargs (remaining dictionary of keyword arguments, `optional`):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                ``output_attentions=True``).

                - To update the question_encoder configuration, use the prefix `question_encoder_` for each
                  configuration parameter.
                - To update the generator configuration, use the prefix `generator_` for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a :obj:`config` is provided or automatically loaded.

        Example::

            >>> from transformers import RagRetriever, TFRagModel
            >>> # initialize a RAG from two pretrained models.
            >>> model = TFRagModel.from_pretrained_question_encoder_generator('facebook/dpr-question_encoder-single-nq-base', 't5-small')
            >>> # alternatively, initialize from pytorch pretrained models can also be done
            >>> model = TFRagModel.from_pretrained_question_encoder_generator('facebook/dpr-question_encoder-single-nq-base', "facebook/bart-base", generator_from_pt=True, question_encoder_from_pt=True)

            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./rag")

            >>> # load retriever
            >>> retriever = RagRetriever.from_pretrained(PATH, index_name="exact", use_dummy_dataset=True)
            >>> # load fine-tuned model with retriever
            >>> model = TFRagModel.from_pretrained("./rag", retriever=retriever)
        """
        ...
    


RAG_START_DOCSTRING = ...
RAG_FORWARD_INPUTS_DOCSTRING = ...
@add_start_docstrings_to_model_forward(RAG_START_DOCSTRING)
class TFRagModel(TFRagPreTrainedModel):
    load_weight_prefix = ...
    def __init__(self, config: Optional[PretrainedConfig] = ..., question_encoder: Optional[TFPreTrainedModel] = ..., generator: Optional[TFPreTrainedModel] = ..., retriever: Optional = ..., load_weight_prefix: Optional[str] = ..., **kwargs) -> None:
        ...
    
    def set_retriever(self, retriever: RagRetriever): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFRetrievAugLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., encoder_outputs=..., decoder_input_ids=..., decoder_attention_mask=..., past_key_values=..., doc_scores=..., context_input_ids=..., context_attention_mask=..., use_cache=..., output_attentions=..., output_hidden_states=..., output_retrieved=..., n_docs=..., return_dict=..., training=..., **kwargs):
        r"""
        Returns:

        Example::

            >>> from transformers import RagTokenizer, RagRetriever, RagModel
            >>> import torch

            >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
            >>> retriever = RagRetriever.from_pretrained("facebook/rag-token-base", index_name="exact", use_dummy_dataset=True)
            >>> # initialize with RagRetriever to do everything in one forward call
            >>> model = TFRagModel.from_pretrained("facebook/rag-token-base", retriever=retriever, from_pt=True)

            >>> input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="tf")
            >>> input_ids = input_dict["input_ids"]
            >>> outputs = model(input_ids)

        """
        ...
    


@add_start_docstrings_to_model_forward("""
    A TF RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.
    """, RAG_START_DOCSTRING)
class TFRagTokenForGeneration(TFRagPreTrainedModel, TFCausalLanguageModelingLoss):
    load_weight_prefix = ...
    def __init__(self, config: Optional[PretrainedConfig] = ..., question_encoder: Optional[TFPreTrainedModel] = ..., generator: Optional[TFPreTrainedModel] = ..., retriever: Optional = ..., **kwargs) -> None:
        ...
    
    def set_retriever(self, retriever: RagRetriever): # -> None:
        ...
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, doc_scores, n_docs=..., **kwargs) -> Dict:
        ...
    
    @property
    def retriever(self): # -> RagRetriever:
        ...
    
    @property
    def generator(self): # -> TFPreTrainedModel | None:
        ...
    
    @property
    def question_encoder(self): # -> TFPreTrainedModel | None:
        ...
    
    def marginalize(self, seq_logits, doc_scores, n_docs=...):
        ...
    
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFRetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., encoder_outputs=..., past_key_values=..., doc_scores=..., context_input_ids=..., context_attention_mask=..., use_cache=..., output_attentions=..., output_hidden_states=..., output_retrieved=..., n_docs=..., do_marginalize=..., labels=..., reduce_loss=..., return_dict=..., training=..., **kwargs): # -> TFRetrievAugLMMarginOutput:
        r"""
        do_marginalize (:obj:`bool`, `optional`):
            If :obj:`True`, the logits are marginalized over all documents by making use of
            ``torch.nn.functional.log_softmax``.
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss according to Rag-Token model formulation See
            https://arxiv.org/pdf/2005.11401.pdf Section 2.1 for details about Rag-Token formulation. Indices should be
            in ``[0, ..., config.vocab_size - 1]``.
        reduce_loss (:obj:`bool`, `optional`):
            Only relevant if ``labels`` is passed. If :obj:`True`, the NLL loss is reduced using the ``tf.Tensor.sum``
            operation.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Legacy dictionary, which is required so that model can use `generate()` function.

        Returns:

        Example::

            >>> from transformers import RagTokenizer, RagRetriever, TFRagTokenForGeneration

            >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
            >>> retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
            >>> # initialize with RagRetriever to do everything in one forward call
            >>> model = TFRagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever, from_pt=True)

            >>> input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="tf")
            >>> outputs = model(input_dict, output_retrieved=True)

            >>> # or use retriever separately
            >>> # 1. Encode
            >>> input_ids = input_dict["input_ids"]
            >>> question_hidden_states = model.question_encoder(input_ids)[0]
            >>> # 2. Retrieve
            >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.numpy(), return_tensors="tf")
            >>> doc_scores = tf.squeeze(tf.matmul(tf.expand_dims(question_hidden_states, axis=1), docs_dict["retrieved_doc_embeds"], transpose_b=True), axis=1)
            >>> # 3. Forward to generator
            >>> outputs = model(inputs=None, context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores, decoder_input_ids=input_dict["labels"])

            >>> # or directly generate
            >>> generated = model.generate(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores)
            >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        """
        ...
    
    def generate(self, input_ids: Optional[tf.Tensor] = ..., attention_mask: Optional[tf.Tensor] = ..., context_input_ids=..., context_attention_mask=..., doc_scores=..., max_length=..., min_length=..., early_stopping=..., use_cache=..., num_beams=..., bos_token_id=..., pad_token_id=..., eos_token_id=..., length_penalty=..., no_repeat_ngram_size=..., bad_words_ids=..., num_return_sequences=..., decoder_start_token_id=..., n_docs=..., output_scores=..., output_attentions=..., output_hidden_states=..., return_dict_in_generate=..., **model_kwargs): # -> TFBeamSearchOutput | TFBeamSampleOutput | TFGreedySearchOutput | TFSampleOutput:
        """
        Implements TFRAG token decoding.

        Args:
            input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`input_ids` is not passed, then
                :obj:`context_input_ids` has to be provided.
            attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            context_input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Input IDs post-processed from the retrieved documents and the question encoder :obj:`input_ids` by the
                retriever.

                If the model has is not initialized with a ``retriever``, :obj:`context_input_ids` has to be provided
                to the forward pass. :obj:`context_input_ids` are returned by
                :meth:`~transformers.RagRetriever.__call__`.
            context_attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by
                the retriever.

                If the model has is not initialized with a ``retriever``, :obj:`context_input_ids` has to be provided
                to the forward pass. :obj:`context_input_ids` are returned by
                :meth:`~transformers.RagRetriever.__call__`.
            doc_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size, config.n_docs)`):
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
            bad_words_ids(:obj:`List[int]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the ``generator``'s
                `:func:`~transformers.generation_utils.GenerationMixin.generate` function, where we set
                ``num_return_sequences`` to :obj:`num_beams`.
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            n_docs (:obj:`int`, `optional`, defaults to :obj:`config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            model_specific_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.

        Return:
            :obj:`tf.Tensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
            batches finished early due to the :obj:`eos_token_id`.
        """
        ...
    
    def get_input_embeddings(self):
        ...
    
    def get_output_embeddings(self): # -> None:
        ...
    
    def shift_tokens_right(self, input_ids, start_token_id=...):
        """Shift input ids one token to the right, and pad with start_token_id"""
        ...
    
    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=..., epsilon=..., n_docs=...):
        ...
    
    def compute_loss(self, labels, y_pred, smooth_epsilon=..., from_logits=..., reduce_loss=...):
        """CrossEntropyLoss that ignores pad tokens"""
        ...
    


@add_start_docstrings_to_model_forward("""
    A TF RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.
    """, RAG_START_DOCSTRING)
class TFRagSequenceForGeneration(TFRagPreTrainedModel, TFCausalLanguageModelingLoss):
    load_weight_prefix = ...
    def __init__(self, config: Optional[PretrainedConfig] = ..., question_encoder: Optional[TFPreTrainedModel] = ..., generator: Optional[TFPreTrainedModel] = ..., retriever: Optional = ..., **kwargs) -> None:
        ...
    
    def set_retriever(self, retriever: RagRetriever): # -> None:
        ...
    
    @property
    def retriever(self): # -> RagRetriever:
        ...
    
    @property
    def generator(self): # -> TFPreTrainedModel | None:
        ...
    
    @property
    def question_encoder(self): # -> TFPreTrainedModel | None:
        ...
    
    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFRetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., encoder_outputs=..., past_key_values=..., doc_scores=..., context_input_ids=..., context_attention_mask=..., use_cache=..., output_attentions=..., output_hidden_states=..., output_retrieved=..., n_docs=..., exclude_bos_score=..., labels=..., reduce_loss=..., return_dict=..., training=..., **kwargs): # -> TFRetrievAugLMMarginOutput:
        r"""
        exclude_bos_score (:obj:`bool`, `optional`):
            Only relevant if ``labels`` is passed. If :obj:`True`, the score of the BOS token is disregarded when
            computing the loss.
        labels (:obj:`tf.Tensor` or :obj:`np.ndarray` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss according to Rag-Sequence model formulation See
            https://arxiv.org/pdf/2005.11401.pdf Section 2.1 for details about Rag-Sequence formulation. Indices should
            be in ``[0, ..., config.vocab_size - 1]``.
        reduce_loss (:obj:`bool`, `optional`):
            Only relevant if ``labels`` is passed. If :obj:`True`, the NLL loss is reduced using the ``tf.Tensor.sum``
            operation.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Legacy dictionary, which is required so that model can use `generate()` function.

        Returns:

        Example::

            >>> from transformers import RagTokenizer, RagRetriever, TFRagSequenceForGeneration

            >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
            >>> retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
            >>> # initialize with RagRetriever to do everything in one forward call
            >>> model = TFRagRagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever, from_pt=True)

            >>> input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="tf")
            >>> outputs = model(input_dict, output_retrieved=True)

            >>> # or use retriever separately
            >>> # 1. Encode
            >>> input_ids = input_dict["input_ids"]
            >>> question_hidden_states = model.question_encoder(input_ids)[0]
            >>> # 2. Retrieve
            >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.numpy(), return_tensors="tf")
            >>> doc_scores = tf.squeeze(tf.matmul(tf.expand_dims(question_hidden_states, axis=1), docs_dict["retrieved_doc_embeds"], transpose_b=True), axis=1)
            >>> # 3. Forward to generator
            >>> outputs = model(inputs=None, context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores, decoder_input_ids=input_dict["labels"])

            >>> # or directly generate
            >>> generated = model.generate(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores)
            >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        """
        ...
    
    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=..., epsilon=..., exclude_bos_score=..., n_docs=...):
        ...
    
    def generate(self, input_ids: Optional[tf.Tensor] = ..., attention_mask: Optional[tf.Tensor] = ..., context_input_ids=..., context_attention_mask=..., doc_scores=..., do_deduplication=..., num_return_sequences=..., num_beams=..., n_docs=..., **model_kwargs):
        """
        Implements RAG sequence "thorough" decoding. Read the
        :meth:`~transformers.generation_utils.GenerationMixin.generate`` documentation for more information on how to
        set other generate input parameters

        Args:
            input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`input_ids` is not passed, then
                :obj:`context_input_ids` has to be provided.
            attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``: - 1
                for tokens that are **not masked**, - 0 for tokens that are **masked**. `What are attention masks?
                <../glossary.html#attention-mask>`__
            context_input_ids (:obj:`tf.Tensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Input IDs post-processed from the retrieved documents and the question encoder input_ids by the
                retriever.
            context_attention_mask (:obj:`tf.Tensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by
                the retriever. If the model has is not initialized with a ``retriever`` or ``input_ids`` is not given,
                :obj:`context_input_ids` and :obj:`context_attention_mask` have to be provided to the forward pass.
                They are returned by :meth:`~transformers.RagRetriever.__call__`.
            doc_scores (:obj:`tf.Tensor` of shape :obj:`(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
                :obj:`question_encoder_last_hidden_state`. If the model has is not initialized with a ``retriever`` or
                ``input_ids`` is not given, :obj:`doc_scores` has to be provided to the forward pass. :obj:`doc_scores`
                are returned by :meth:`~transformers.RagRetriever.__call__`.
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
                Additional kwargs will be passed to :meth:`~transformers.generation_utils.GenerationMixin.generate`

        Return:
            :obj:`tf.Tensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence length) is either equal to :obj:`max_length` or shorter if all
            batches finished early due to the :obj:`eos_token_id`.
        """
        ...
    


