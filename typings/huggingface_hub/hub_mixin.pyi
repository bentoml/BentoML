from typing import Dict, Optional, Union
from .file_download import is_torch_available

if is_torch_available(): ...
logger = ...

class ModelHubMixin:
    def save_pretrained(
        self,
        save_directory: str,
        config: Optional[dict] = ...,
        push_to_hub: bool = ...,
        **kwargs
    ): ...
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str],
        force_download: bool = ...,
        resume_download: bool = ...,
        proxies: Dict = ...,
        use_auth_token: Optional[str] = ...,
        cache_dir: Optional[str] = ...,
        local_files_only: bool = ...,
        **model_kwargs
    ): ...
    def push_to_hub(
        self,
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
    ) -> str: ...

class PyTorchModelHubMixin(ModelHubMixin):
    def __init__(self, *args, **kwargs) -> None: ...
