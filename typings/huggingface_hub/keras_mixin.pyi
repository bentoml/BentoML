from typing import Any, Dict, Optional, Union
from huggingface_hub import ModelHubMixin

logger = ...

def save_pretrained_keras(
    model, save_directory: str, config: Optional[Dict[str, Any]] = ...
): ...
def from_pretrained_keras(*args, **kwargs): ...
def push_to_hub_keras(
    model,
    repo_path_or_name: Optional[str] = ...,
    repo_url: Optional[str] = ...,
    commit_message: Optional[str] = ...,
    organization: Optional[str] = ...,
    private: Optional[bool] = ...,
    api_endpoint: Optional[str] = ...,
    use_auth_token: Optional[Union[bool, str]] = ...,
    git_user: Optional[str] = ...,
    git_email: Optional[str] = ...,
    config: Optional[dict] = ...,
): ...

class KerasModelHubMixin(ModelHubMixin):
    def __init__(self, *args, **kwargs) -> None: ...
