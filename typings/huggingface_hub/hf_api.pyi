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

def repo_type_and_id_from_hf_id(hf_id: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Returns the repo type and ID from a huggingface.co URL linking to a repository

    Args:
        hf_id (``str``):
            An URL or ID of a repository on the HF hub. Accepted values are:
            - https://huggingface.co/<repo_type>/<namespace>/<repo_id>
            - https://huggingface.co/<namespace>/<repo_id>
            - <repo_type>/<namespace>/<repo_id>
            - <namespace>/<repo_id>
            - <repo_id>
    """
    ...

class RepoObj:
    """
    HuggingFace git-based system, data structure that represents a file belonging to the current user.
    """
    def __init__(self, **kwargs: Any) -> None:
        ...

    def __repr__(self) -> str:
        ...



class ModelFile:
    """
    Data structure that represents a public file inside a model, accessible from huggingface.co
    """
    def __init__(self, rfilename: str, **kwargs: Any) -> None:
        ...

    def __repr__(self) -> str:
        ...



class DatasetFile:
    """
    Data structure that represents a public file inside a dataset, accessible from huggingface.co
    """
    def __init__(self, rfilename: str, **kwargs: Any) -> None:
        ...

    def __repr__(self) -> str:
        ...



class ModelInfo:
    """
    Info about a public model accessible from huggingface.co
    """
    def __init__(self, modelId: Optional[str] = ..., sha: Optional[str] = ..., lastModified: Optional[str] = ..., tags: List[str] = ..., pipeline_tag: Optional[str] = ..., siblings: Optional[List[Dict[str, Any]]] = ..., config: Optional[Dict[str, Any]] = ..., **kwargs: Any) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...



class DatasetInfo:
    """
    Info about a public dataset accessible from huggingface.co
    """
    def __init__(self, id: Optional[str] = ..., lastModified: Optional[str] = ..., tags: List[str] = ..., siblings: Optional[List[Dict[str, Any]]] = ..., private: Optional[bool] = ..., author: Optional[str] = ..., description: Optional[str] = ..., citation: Optional[str] = ..., card_data: Optional[dict] = ..., **kwargs) -> None:
        ...

    def __repr__(self): # -> str:
        ...

    def __str__(self) -> str:
        ...



class MetricInfo:
    """
    Info about a public metric accessible from huggingface.co
    """
    def __init__(self, id: Optional[str] = ..., description: Optional[str] = ..., citation: Optional[str] = ..., **kwargs: Any) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...



def write_to_credential_store(username: str, password: str) -> None:
    ...

def read_from_credential_store(username: Optional[str]=...) -> Tuple[Optional[str], Optional[str]]:
    """
    Reads the credential store relative to huggingface.co. If no `username` is specified, will read the first
    entry for huggingface.co, otherwise will read the entry corresponding to the username specified.

    The username returned will be all lowercase.
    """
    ...

def erase_from_credential_store(username: Optional[str]=...) -> None:
    """
    Erases the credential store relative to huggingface.co. If no `username` is specified, will erase the first
    entry for huggingface.co, otherwise will erase the entry corresponding to the username specified.
    """
    ...

class HfApi:
    def __init__(self, endpoint: str=...) -> None:
        ...

    def login(self, username: str, password: str) -> str:
        """
        Call HF API to sign in a user and get a token if credentials are valid.

        Outputs: token if credentials are valid

        Throws: requests.exceptions.HTTPError if credentials are invalid
        """
        ...

    def whoami(self, token: Optional[str] = ...) -> Dict[str, Any]:
        """
        Call HF API to know "whoami".

        Args:
            token (``str``, `optional`):
                Hugging Face token. Will default to the locally saved token if not provided.
        """
        ...

    def logout(self, token: Optional[str] = ...) -> None:
        """
        Call HF API to log out.

        Args:
            token (``str``, `optional`):
                Hugging Face token. Will default to the locally saved token if not provided.
        """
        ...

    def list_models(self, filter: Union[str, Iterable[str], None] = ..., sort: Union[Literal["lastModified"], str, None] = ..., direction: Optional[Literal[-1]] = ..., limit: Optional[int] = ..., full: Optional[bool] = ..., fetch_config: Optional[bool] = ...) -> List[ModelInfo]:
        """
        Get the public list of all the models on huggingface.co

        Args:
            filter (:obj:`str` or :class:`Iterable`, `optional`):
                A string which can be used to identify models on the hub by their tags.
                Example usage:

                    >>> from huggingface_hub import HfApi
                    >>> api = HfApi()

                    >>> # List all models
                    >>> api.list_models()

                    >>> # List only the text classification models
                    >>> api.list_models(filter="text-classification")

                    >>> # List only the russian models compatible with pytorch
                    >>> api.list_models(filter=("ru", "pytorch"))

                    >>> # List only the models trained on the "common_voice" dataset
                    >>> api.list_models(filter="dataset:common_voice")

                    >>> # List only the models from the AllenNLP library
                    >>> api.list_models(filter="allennlp")
            sort (:obj:`Literal["lastModified"]` or :obj:`str`, `optional`):
                The key with which to sort the resulting models. Possible values are the properties of the `ModelInfo`
                class.
            direction (:obj:`Literal[-1]` or :obj:`int`, `optional`):
                Direction in which to sort. The value `-1` sorts by descending order while all other values
                sort by ascending order.
            limit (:obj:`int`, `optional`):
                The limit on the number of models fetched. Leaving this option to `None` fetches all models.
            full (:obj:`bool`, `optional`):
                Whether to fetch all model data, including the `lastModified`, the `sha`, the files and the `tags`.
                This is set to `True` by default when using a filter.
            fetch_config (:obj:`bool`, `optional`):
                Whether to fetch the model configs as well. This is not included in `full` due to its size.

        """
        ...

    def model_list(self) -> List[ModelInfo]:
        """
        Deprecated method name, renamed to `list_models`.

        Get the public list of all the models on huggingface.co
        """
        ...

    def list_datasets(self, filter: Union[str, Iterable[str], None] = ..., sort: Union[Literal["lastModified"], str, None] = ..., direction: Optional[Literal[-1]] = ..., limit: Optional[int] = ..., full: Optional[bool] = ...) -> List[DatasetInfo]:
        """
        Get the public list of all the datasets on huggingface.co

        Args:
            filter (:obj:`str` or :class:`Iterable`, `optional`):
                A string which can be used to identify datasets on the hub by their tags.
                Example usage:

                    >>> from huggingface_hub import HfApi
                    >>> api = HfApi()

                    >>> # List all datasets
                    >>> api.list_datasets()

                    >>> # List only the text classification datasets
                    >>> api.list_datasets(filter="task_categories:text-classification")

                    >>> # List only the datasets in russian for language modeling
                    >>> api.list_datasets(filter=("languages:ru", "task_ids:language-modeling"))
            sort (:obj:`Literal["lastModified"]` or :obj:`str`, `optional`):
                The key with which to sort the resulting datasets. Possible values are the properties of the `DatasetInfo`
                class.
            direction (:obj:`Literal[-1]` or :obj:`int`, `optional`):
                Direction in which to sort. The value `-1` sorts by descending order while all other values
                sort by ascending order.
            limit (:obj:`int`, `optional`):
                The limit on the number of datasets fetched. Leaving this option to `None` fetches all datasets.
            full (:obj:`bool`, `optional`):
                Whether to fetch all dataset data, including the `lastModified` and the `card_data`.

        """
        ...

    def list_metrics(self) -> List[MetricInfo]:
        """
        Get the public list of all the metrics on huggingface.co
        """
        ...

    def model_info(self, repo_id: str, revision: Optional[str] = ..., token: Optional[str] = ..., timeout: Optional[float] = ...) -> ModelInfo:
        """
        Get info on one specific model on huggingface.co

        Model can be private if you pass an acceptable token or are logged in.
        """
        ...

    def list_repo_files(self, repo_id: str, revision: Optional[str] = ..., repo_type: Optional[str] = ..., token: Optional[str] = ..., timeout: Optional[float] = ...) -> List[str]:
        """
        Get the list of files in a given repo.
        """
        ...

    def list_repos_objs(self, token: Optional[str] = ..., organization: Optional[str] = ...) -> List[RepoObj]:
        """
        HuggingFace git-based system, used for models, datasets, and spaces.

        Call HF API to list all stored files for user (or one of their organizations).
        """
        ...

    def dataset_info(self, repo_id: str, revision: Optional[str] = ..., token: Optional[str] = ..., timeout: Optional[float] = ...) -> DatasetInfo:
        """
        Get info on one specific dataset on huggingface.co

        Dataset can be private if you pass an acceptable token.
        """
        ...

    def create_repo(self, name: str, token: Optional[str] = ..., organization: Optional[str] = ..., private: Optional[bool] = ..., repo_type: Optional[str] = ..., exist_ok: bool=..., lfsmultipartthresh: Optional[int] = ..., space_sdk: Optional[str] = ...) -> str:
        """
        HuggingFace git-based system, used for models, datasets, and spaces.

        Call HF API to create a whole repo.

        Params:
            private: Whether the model repo should be private (requires a paid huggingface.co account)

            repo_type: Set to "dataset" or "space" if creating a dataset or space, default is model

            exist_ok: Do not raise an error if repo already exists

            lfsmultipartthresh: Optional: internal param for testing purposes.

            space_sdk: Choice of SDK to use if repo_type is "space". Can be "streamlit", "gradio", or "static".

        Returns:
            URL to the newly created repo.
        """
        ...

    def delete_repo(self, name: str, token: Optional[str] = ..., organization: Optional[str] = ..., repo_type: Optional[str] = ...) -> None:
        """
        HuggingFace git-based system, used for models, datasets, and spaces.

        Call HF API to delete a whole repo.

        CAUTION(this is irreversible).
        """
        ...

    def update_repo_visibility(self, name: str, private: bool, token: Optional[str] = ..., organization: Optional[str] = ..., repo_type: Optional[str] = ...) -> Dict[str, bool]:
        """
        Update the visibility setting of a repository.
        """
        ...

    def upload_file(self, path_or_fileobj: Union[str, bytes, TextIO], path_in_repo: str, repo_id: str, token: Optional[str] = ..., repo_type: Optional[str] = ..., revision: Optional[str] = ..., identical_ok: bool = ...) -> str:
        """
        Upload a local file (up to 5GB) to the given repo. The upload is done through a HTTP post request, and
        doesn't require git or git-lfs to be installed.

        Params:
            path_or_fileobj (``str``, ``bytes``, or ``IO``):
                Path to a file on the local machine or binary data stream / fileobj / buffer.

            path_in_repo (``str``):
                Relative filepath in the repo, for example: :obj:`"checkpoints/1fec34a/weights.bin"`

            repo_id (``str``):
                The repository to which the file will be uploaded, for example: :obj:`"username/custom_transformers"`

            token (``str``):
                Authentication token, obtained with :function:`HfApi.login` method. Will default to the stored token.

            repo_type (``str``, Optional):
                Set to :obj:`"dataset"` or :obj:`"space"` if uploading to a dataset or space, :obj:`None` if uploading to a model. Default is :obj:`None`.

            revision (``str``, Optional):
                The git revision to commit from. Defaults to the :obj:`"main"` branch.

            identical_ok (``bool``, defaults to ``True``):
                When set to false, will raise an HTTPError when the file you're trying to upload already exists on the hub
                and its content did not change.

        Returns:
            ``str``: The URL to visualize the uploaded file on the hub

        Raises:
            :class:`ValueError`: if some parameter value is invalid

            :class:`requests.HTTPError`: if the HuggingFace API returned an error

        Examples:
            >>> with open("./local/filepath", "rb") as fobj:
            ...     upload_file(
            ...         path_or_fileobj=fileobj,
            ...         path_in_repo="remote/file/path.h5",
            ...         repo_id="username/my-dataset",
            ...         repo_type="datasets",
            ...         token="my_token",
            ...    )
            "https://huggingface.co/datasets/username/my-dataset/blob/main/remote/file/path.h5"

            >>> upload_file(
            ...     path_or_fileobj=".\\\\local\\\\file\\\\path",
            ...     path_in_repo="remote/file/path.h5",
            ...     repo_id="username/my-model",
            ...     token="my_token",
            ... )
            "https://huggingface.co/username/my-model/blob/main/remote/file/path.h5"


        """
        ...

    def delete_file(self, path_in_repo: str, repo_id: str, token: Optional[str] = ..., repo_type: Optional[str] = ..., revision: Optional[str] = ...) -> None:
        """
        Deletes a file in the given repo.

        Params:
            path_in_repo (``str``):
                Relative filepath in the repo, for example: :obj:`"checkpoints/1fec34a/weights.bin"`

            repo_id (``str``):
                The repository from which the file will be deleted, for example: :obj:`"username/custom_transformers"`

            token (``str``):
                Authentication token, obtained with :function:`HfApi.login` method. Will default to the stored token.

            repo_type (``str``, Optional):
                Set to :obj:`"dataset"` or :obj:`"space"` if the file is in a dataset or space repository, :obj:`None` if in a model. Default is :obj:`None`.

            revision (``str``, Optional):
                The git revision to commit from. Defaults to the :obj:`"main"` branch.

        Raises:
            :class:`ValueError`: if some parameter value is invalid

            :class:`requests.HTTPError`: if the HuggingFace API returned an error

        """
        ...

    def get_full_repo_name(self, model_id: str, organization: Optional[str] = ..., token: Optional[str] = ...) -> str:
        """
        Returns the repository name for a given model ID and optional organization.

        Args:
            model_id (``str``):
                The name of the model.
            organization (``str``, `optional`):
                If passed, the repository name will be in the organization namespace instead of the
                user namespace.
            token (``str``, `optional`):
                The Hugging Face authentication token

        Returns:
            ``str``: The repository name in the user's namespace ({username}/{model_id}) if no
            organization is passed, and under the organization namespace ({organization}/{model_id})
            otherwise.
        """
        ...



class HfFolder:
    path_token: os.PathLike[str] = ...
    @classmethod
    def save_token(cls, token: str) -> None:
        """
        Save token, creating folder as needed.
        """
        ...

    @classmethod
    def get_token(cls) -> Optional[str]:
        """
        Get token or None if not existent.
        """
        ...

    @classmethod
    def delete_token(cls) -> None:
        """
        Delete token. Do not fail if token does not exist.
        """
        ...

