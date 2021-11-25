

from ..file_utils import add_end_docstrings
from .base import PIPELINE_INIT_ARGS, Pipeline

@add_end_docstrings(PIPELINE_INIT_ARGS)
class TextGenerationPipeline(Pipeline):
    """
    Language generation pipeline using any :obj:`ModelWithLMHead`. This pipeline predicts the words that will follow a
    specified text prompt.

    This language generation pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"text-generation"`.

    The models that this pipeline can use are models that have been trained with an autoregressive language modeling
    objective, which includes the uni-directional models in the library (e.g. gpt2). See the list of available models
    on `huggingface.co/models <https://huggingface.co/models?filter=causal-lm>`__.
    """
    XL_PREFIX = ...
    ALLOWED_MODELS = ...
    def __init__(self, *args, return_full_text=..., **kwargs) -> None:
        ...
    
    def __call__(self, text_inputs, return_tensors=..., return_text=..., return_full_text=..., clean_up_tokenization_spaces=..., prefix=..., **generate_kwargs):
        """
        Complete the prompt(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several prompts (or one list of prompts) to complete.
            return_tensors (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to include the decoded texts in the outputs.
            return_full_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to :obj:`False` only added text is returned, otherwise the full text is returned Only meaningful
                if `return_text` is set to True.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            prefix (:obj:`str`, `optional`):
                Prefix added to prompt.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a dictionary with the following keys:

            - **generated_text** (:obj:`str`, present when ``return_text=True``) -- The generated text.
            - **generated_token_ids** (:obj:`torch.Tensor` or :obj:`tf.Tensor`, present when ``return_tensors=True``)
              -- The token ids of the generated text.
        """
        ...
    


