

import io
from typing import Dict, List, Optional, Tuple

ENDPOINT = ...
class RepoObj:
    """
    HuggingFace git-based system, data structure that represents a file belonging to the current user.
    """
    def __init__(self, filename: str, lastModified: str, commit: str, size: int, **kwargs) -> None:
        ...
    


class ModelSibling:
    """
    Data structure that represents a public file inside a model, accessible from huggingface.co
    """
    def __init__(self, rfilename: str, **kwargs) -> None:
        ...
    


class ModelInfo:
    """
    Info about a public model accessible from huggingface.co
    """
    def __init__(self, modelId: Optional[str] = ..., tags: List[str] = ..., pipeline_tag: Optional[str] = ..., siblings: Optional[List[Dict]] = ..., **kwargs) -> None:
        ...
    


class HfApi:
    def __init__(self, endpoint=...) -> None:
        ...
    
    def login(self, username: str, password: str) -> str:
        """
        Call HF API to sign in a user and get a token if credentials are valid.

        Outputs: token if credentials are valid

        Throws: requests.exceptions.HTTPError if credentials are invalid
        """
        ...
    
    def whoami(self, token: str) -> Tuple[str, List[str]]:
        """
        Call HF API to know "whoami"
        """
        ...
    
    def logout(self, token: str) -> None:
        """
        Call HF API to log out.
        """
        ...
    
    def model_list(self) -> List[ModelInfo]:
        """
        Get the public list of all the models on huggingface.co
        """
        ...
    
    def list_repos_objs(self, token: str, organization: Optional[str] = ...) -> List[RepoObj]:
        """
        HuggingFace git-based system, used for models.

        Call HF API to list all stored files for user (or one of their organizations).
        """
        ...
    
    def create_repo(self, token: str, name: str, organization: Optional[str] = ..., private: Optional[bool] = ..., exist_ok=..., lfsmultipartthresh: Optional[int] = ...) -> str:
        """
        HuggingFace git-based system, used for models.

        Call HF API to create a whole repo.

        Params:
            private: Whether the model repo should be private (requires a paid huggingface.co account)

            exist_ok: Do not raise an error if repo already exists

            lfsmultipartthresh: Optional: internal param for testing purposes.
        """
        ...
    
    def delete_repo(self, token: str, name: str, organization: Optional[str] = ...):
        """
        HuggingFace git-based system, used for models.

        Call HF API to delete a whole repo.

        CAUTION(this is irreversible).
        """
        ...
    


class TqdmProgressFileReader:
    """
    Wrap an io.BufferedReader `f` (such as the output of `open(…, "rb")`) and override `f.read()` so as to display a
    tqdm progress bar.

    see github.com/huggingface/transformers/pull/2078#discussion_r354739608 for implementation details.
    """
    def __init__(self, f: io.BufferedReader) -> None:
        ...
    
    def close(self):
        ...
    


class HfFolder:
    path_token = ...
    @classmethod
    def save_token(cls, token):
        """
        Save token, creating folder as needed.
        """
        ...
    
    @classmethod
    def get_token(cls):
        """
        Get token or None if not existent.
        """
        ...
    
    @classmethod
    def delete_token(cls):
        """
        Delete token. Do not fail if token does not exist.
        """
        ...
    


