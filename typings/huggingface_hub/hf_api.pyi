import os
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Pattern,
    TextIO,
    Tuple,
    Union,
)

REMOTE_FILEPATH_REGEX: Pattern[str] = ...

def repo_type_and_id_from_hf_id(
    hf_id: str,
) -> Tuple[Optional[str], Optional[str], str]: ...

class RepoObj:
    def __init__(self, **kwargs: Any) -> None: ...
    def __repr__(self) -> str: ...

class ModelFile:
    def __init__(self, rfilename: str, **kwargs: Any) -> None: ...
    def __repr__(self) -> str: ...

class DatasetFile:
    def __init__(self, rfilename: str, **kwargs: Any) -> None: ...
    def __repr__(self) -> str: ...

class ModelInfo:
    def __init__(
        self,
        modelId: Optional[str] = ...,
        sha: Optional[str] = ...,
        lastModified: Optional[str] = ...,
        tags: List[str] = ...,
        pipeline_tag: Optional[str] = ...,
        siblings: Optional[List[Dict[str, Any]]] = ...,
        config: Optional[Dict[str, Any]] = ...,
        **kwargs: Any
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class DatasetInfo:
    def __init__(
        self,
        id: Optional[str] = ...,
        lastModified: Optional[str] = ...,
        tags: List[str] = ...,
        siblings: Optional[List[Dict[str, Any]]] = ...,
        private: Optional[bool] = ...,
        author: Optional[str] = ...,
        description: Optional[str] = ...,
        citation: Optional[str] = ...,
        card_data: Optional[dict] = ...,
        **kwargs
    ) -> None: ...
    def __repr__(self): ...
    def __str__(self) -> str: ...

class MetricInfo:
    def __init__(
        self,
        id: Optional[str] = ...,
        description: Optional[str] = ...,
        citation: Optional[str] = ...,
        **kwargs: Any
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

def write_to_credential_store(username: str, word: str) -> None: ...
def read_from_credential_store(
    username: Optional[str] = ...,
) -> Tuple[Optional[str], Optional[str]]: ...
def erase_from_credential_store(username: Optional[str] = ...) -> None: ...

class HfApi:
    def __init__(self, endpoint: str = ...) -> None: ...
    def login(self, username: str, word: str) -> str: ...
    def whoami(self, token: Optional[str] = ...) -> Dict[str, Any]: ...
    def logout(self, token: Optional[str] = ...) -> None: ...
    def list_models(
        self,
        filter: Union[str, Iterable[str], None] = ...,
        sort: Union[Literal["lastModified"], str, None] = ...,
        direction: Optional[Literal[-1]] = ...,
        limit: Optional[int] = ...,
        full: Optional[bool] = ...,
        fetch_config: Optional[bool] = ...,
    ) -> List[ModelInfo]: ...
    def model_list(self) -> List[ModelInfo]: ...
    def list_datasets(
        self,
        filter: Union[str, Iterable[str], None] = ...,
        sort: Union[Literal["lastModified"], str, None] = ...,
        direction: Optional[Literal[-1]] = ...,
        limit: Optional[int] = ...,
        full: Optional[bool] = ...,
    ) -> List[DatasetInfo]: ...
    def list_metrics(self) -> List[MetricInfo]: ...
    def model_info(
        self,
        repo_id: str,
        revision: Optional[str] = ...,
        token: Optional[str] = ...,
        timeout: Optional[float] = ...,
    ) -> ModelInfo: ...
    def list_repo_files(
        self,
        repo_id: str,
        revision: Optional[str] = ...,
        repo_type: Optional[str] = ...,
        token: Optional[str] = ...,
        timeout: Optional[float] = ...,
    ) -> List[str]: ...
    def list_repos_objs(
        self, token: Optional[str] = ..., organization: Optional[str] = ...
    ) -> List[RepoObj]: ...
    def dataset_info(
        self,
        repo_id: str,
        revision: Optional[str] = ...,
        token: Optional[str] = ...,
        timeout: Optional[float] = ...,
    ) -> DatasetInfo: ...
    def create_repo(
        self,
        name: str,
        token: Optional[str] = ...,
        organization: Optional[str] = ...,
        private: Optional[bool] = ...,
        repo_type: Optional[str] = ...,
        exist_ok: bool = ...,
        lfsmultipartthresh: Optional[int] = ...,
        space_sdk: Optional[str] = ...,
    ) -> str: ...
    def delete_repo(
        self,
        name: str,
        token: Optional[str] = ...,
        organization: Optional[str] = ...,
        repo_type: Optional[str] = ...,
    ) -> None: ...
    def update_repo_visibility(
        self,
        name: str,
        private: bool,
        token: Optional[str] = ...,
        organization: Optional[str] = ...,
        repo_type: Optional[str] = ...,
    ) -> Dict[str, bool]: ...
    def upload_file(
        self,
        path_or_fileobj: Union[str, bytes, TextIO],
        path_in_repo: str,
        repo_id: str,
        token: Optional[str] = ...,
        repo_type: Optional[str] = ...,
        revision: Optional[str] = ...,
        identical_ok: bool = ...,
    ) -> str: ...
    def delete_file(
        self,
        path_in_repo: str,
        repo_id: str,
        token: Optional[str] = ...,
        repo_type: Optional[str] = ...,
        revision: Optional[str] = ...,
    ) -> None: ...
    def get_full_repo_name(
        self,
        model_id: str,
        organization: Optional[str] = ...,
        token: Optional[str] = ...,
    ) -> str: ...

class HfFolder:
    path_token: os.PathLike[str] = ...
    @classmethod
    def save_token(cls, token: str) -> None: ...
    @classmethod
    def get_token(cls) -> Optional[str]: ...
    @classmethod
    def delete_token(cls) -> None: ...
