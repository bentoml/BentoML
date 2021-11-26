from typing import Dict, List, Optional, Union

logger = ...
ENDPOINT = ...
ALL_TASKS = ...
class InferenceApi:
    """Client to configure requests and make calls to the HuggingFace Inference API.

    Example:

            >>> from huggingface_hub.inference_api import InferenceApi

            >>> # Mask-fill example
            >>> inference = InferenceApi("bert-base-uncased")
            >>> inference(inputs="The goal of life is [MASK].")
            >>> >> [{'sequence': 'the goal of life is life.', 'score': 0.10933292657136917, 'token': 2166, 'token_str': 'life'}]

            >>> # Question Answering example
            >>> inference = InferenceApi("deepset/roberta-base-squad2")
            >>> inputs = {"question":"What's my name?", "context":"My name is Clara and I live in Berkeley."}
            >>> inference(inputs)
            >>> >> {'score': 0.9326569437980652, 'start': 11, 'end': 16, 'answer': 'Clara'}

            >>> # Zero-shot example
            >>> inference = InferenceApi("typeform/distilbert-base-uncased-mnli")
            >>> inputs = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
            >>> params = {"candidate_labels":["refund", "legal", "faq"]}
            >>> inference(inputs, params)
            >>> >> {'sequence': 'Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!', 'labels': ['refund', 'faq', 'legal'], 'scores': [0.9378499388694763, 0.04914155602455139, 0.013008488342165947]}

            >>> # Overriding configured task
            >>> inference = InferenceApi("bert-base-uncased", task="feature-extraction")
    """
    def __init__(self, repo_id: str, task: Optional[str] = ..., token: Optional[str] = ..., gpu: Optional[bool] = ...) -> None:
        """Inits headers and API call information.

        Args:
            repo_id (``str``): Id of repository (e.g. `user/bert-base-uncased`).
            task (``str``, `optional`, defaults ``None``): Whether to force a task instead of using task specified in the repository.
            token (:obj:`str`, `optional`):
                The API token to use as HTTP bearer authorization. This is not the authentication token.
                You can find the token in https://huggingface.co/settings/token. Alternatively, you can
                find both your organizations and personal API tokens using `HfApi().whoami(token)`.
            gpu (``bool``, `optional`, defaults ``False``): Whether to use GPU instead of CPU for inference(requires Startup plan at least).
        .. note::
            Setting :obj:`token` is required when you want to use a private model.
        """
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def __call__(self, inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = ..., params: Optional[Dict] = ..., data: Optional[bytes] = ...): # -> Any:
        ...
    


