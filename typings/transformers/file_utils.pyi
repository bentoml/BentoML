

import os
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from tempfile import _TemporaryFileWrapper  # type: ignore[reportPrivateUsage]
from types import ModuleType
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Tuple, Union, overload

CONFIG_NAME: str =...
FLAX_WEIGHTS_NAME: str = ...
TF2_WEIGHTS_NAME: str = ...
WEIGHTS_NAME: str = ...

def requires_backends(obj, backends):
    ...

def add_start_docstrings(*docstr):
    ...

def add_start_docstrings_to_model_forward(*docstr):
    ...

def add_end_docstrings(*docstr: str) -> Callable[[Callable[..., Any]], Any]:
    ...

def add_code_sample_docstrings(*docstr, tokenizer_class=..., checkpoint=..., output_type=..., config_class=..., mask=..., model_cls=...):
    ...

def replace_return_docstrings(output_type=..., config_class=...):
    ...

def is_remote_url(url_or_filename):
    ...

def hf_bucket_url(model_id: str, filename: str, subfolder: Optional[str] = ..., revision: Optional[str] = ..., mirror=...) -> str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co-hosted url, redirecting
    to Cloudfront (a Content Delivery Network, or CDN) for large files.

    Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
    bandwidth costs).

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we migrated to a git-based versioning system on huggingface.co, so we now store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object' ETag is:
    its sha1 if stored in git, or its sha256 if stored in git-lfs. Files cached locally from transformers before v3.5.0
    are not shared with those new files, because the cached file's name contains a hash of the url (which changed).
    """
    ...

def url_to_filename(url: str, etag: Optional[str] = ...) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way. If `etag` is specified, append its hash to the url's,
    delimited by a period. If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
    identify it as a HDF5 file (see
    https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    ...

def filename_to_url(filename, cache_dir=...):
    """
    Return the url and etag (which may be ``None``) stored for `filename`. Raise ``EnvironmentError`` if `filename` or
    its stored metadata do not exist.
    """
    ...

def get_cached_models(cache_dir: Union[str, Path] = ...) -> List[Tuple]:
    """
    Returns a list of tuples representing model binaries that are cached locally. Each tuple has shape
    :obj:`(model_url, etag, size_MB)`. Filenames in :obj:`cache_dir` are use to get the metadata for each model, only
    urls ending with `.bin` are added.

    Args:
        cache_dir (:obj:`Union[str, Path]`, `optional`):
            The cache directory to search for models within. Will default to the transformers cache if unset.

    Returns:
        List[Tuple]: List of tuples each with shape :obj:`(model_url, etag, size_MB)`
    """
    ...

def cached_path(url_or_filename, cache_dir=..., force_download=..., proxies=..., resume_download=..., user_agent: Union[Dict, str, None] = ..., extract_compressed_file=..., force_extract=..., use_auth_token: Union[bool, str, None] = ..., local_files_only=...) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path), determine which. If it's a URL, download the file
    and cache it, and return the path to the cached file. If it's already a local path, make sure the file exists and
    then return the path

    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletely received file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        use_auth_token: Optional string or boolean to use as Bearer token for remote files. If True,
            will get token from ~/.huggingface.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and override the folder where it was extracted.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    ...

def define_sagemaker_information():
    ...

def http_user_agent(user_agent: Union[Dict[str, Any], str, None] = ...) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ...

@overload
def http_get(url: str, temp_file: _TemporaryFileWrapper[str], proxies: Optional[str]=..., resume_size: int=..., headers: Optional[Dict[str, str]] = ...) -> None:
    ...
@overload
def http_get(url: str, temp_file: BinaryIO, proxies: Optional[str]=..., resume_size: int=..., headers: Optional[Dict[str, str]] = ...) -> None:
    ...

def get_from_cache(url: str, cache_dir=..., force_download=..., proxies=..., etag_timeout=..., resume_download=..., user_agent: Union[Dict, str, None] = ..., use_auth_token: Union[bool, str, None] = ..., local_files_only=...) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    ...

def get_list_of_files(path_or_repo: Union[str, os.PathLike], revision: Optional[str] = ..., use_auth_token: Optional[Union[bool, str]] = ...) -> List[str]:
    """
    Gets the list of files inside :obj:`path_or_repo`.

    Args:
        path_or_repo (:obj:`str` or :obj:`os.PathLike`):
            Can be either the id of a repo on huggingface.co or a path to a `directory`.
        revision (:obj:`str`, `optional`, defaults to :obj:`"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
            identifier allowed by git.
        use_auth_token (:obj:`str` or `bool`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
            generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).

    Returns:
        :obj:`List[str]`: The list of files available in :obj:`path_or_repo`.
    """
    ...

class cached_property(property):
    """
    Descriptor that mimics @property but caches output in member variable.

    From tensorflow_datasets

    Built-in in functools from Python 3.8.
    """
    def __get__(self, obj, objtype=...):
        ...


def torch_required(func):
    ...

def tf_required(func):
    ...

def is_torch_fx_proxy(x):
    ...

def is_tensor(x):
    """
    Tests if ``x`` is a :obj:`torch.Tensor`, :obj:`tf.Tensor`, obj:`jaxlib.xla_extension.DeviceArray` or
    :obj:`np.ndarray`.
    """
    ...

def to_py_obj(obj):
    """
    Convert a TensorFlow tensor, PyTorch tensor, Numpy array or python list to a python list.
    """
    ...

class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
    a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes. Otherwise behaves like a regular
    python dictionary.

    .. warning::
        You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
        method to convert it to a tuple before.
    """
    def __post_init__(self):
        ...
    
    def __delitem__(self, *args, **kwargs):
        ...
    
    def setdefault(self, *args, **kwargs):
        ...
    
    def pop(self, *args, **kwargs):
        ...
    
    def update(self, *args, **kwargs):
        ...
    
    def __getitem__(self, k):
        ...
    
    def __setattr__(self, name, value):
        ...
    
    def __setitem__(self, key, value):
        ...
    
    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        ...
    


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """
    ...


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the ``padding`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for tab-completion
    in an IDE.
    """
    LONGEST = ...
    MAX_LENGTH = ...
    DO_NOT_PAD = ...


class TensorType(ExplicitEnum):
    """
    Possible values for the ``return_tensors`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for
    tab-completion in an IDE.
    """
    PYTORCH = ...
    TENSORFLOW = ...
    NUMPY = ...
    JAX = ...


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """
    def __init__(self, name, module_file, import_structure, extra_objects=...) -> None:
        ...
    
    def __dir__(self):
        ...
    
    def __getattr__(self, name: str) -> Any:
        ...
    
    def __reduce__(self):
        ...
    


def copy_func(f):
    """Returns a copy of a function f."""
    ...

def is_local_clone(repo_path, repo_url):
    """
    Checks if the folder in `repo_path` is a local clone of `repo_url`.
    """
    ...

class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """
    def push_to_hub(self, repo_path_or_name: Optional[str] = ..., repo_url: Optional[str] = ..., use_temp_dir: bool = ..., commit_message: Optional[str] = ..., organization: Optional[str] = ..., private: Optional[bool] = ..., use_auth_token: Optional[Union[bool, str]] = ...) -> str:
        """
        Upload the {object_files} to the 🤗 Model Hub while synchronizing a local clone of the repo in
        :obj:`repo_path_or_name`.

        Parameters:
            repo_path_or_name (:obj:`str`, `optional`):
                Can either be a repository name for your {object} in the Hub or a path to a local folder (in which case
                the repository will have the name of that local folder). If not specified, will default to the name
                given by :obj:`repo_url` and a local directory with that name will be created.
            repo_url (:obj:`str`, `optional`):
                Specify this in case you want to push to an existing repository in the hub. If unspecified, a new
                repository will be created in your namespace (unless you specify an :obj:`organization`) with
                :obj:`repo_name`.
            use_temp_dir (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clone the distant repo in a temporary directory or in :obj:`repo_path_or_name` inside
                the current working directory. This will slow things down if you are making changes in an existing repo
                since you will need to clone the repo before every push.
            commit_message (:obj:`str`, `optional`):
                Message to commit while pushing. Will default to :obj:`"add {object}"`.
            organization (:obj:`str`, `optional`):
                Organization in which you want to push your {object} (you must be a member of this organization).
            private (:obj:`bool`, `optional`):
                Whether or not the repository created should be private (requires a paying subscription).
            use_auth_token (:obj:`bool` or :obj:`str`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`). Will default to
                :obj:`True` if :obj:`repo_url` is not specified.


        Returns:
            :obj:`str`: The url of the commit of your {object} in the given repository.

        Examples::

            from transformers import {object_class}

            {object} = {object_class}.from_pretrained("bert-base-cased")

            # Push the {object} to your namespace with the name "my-finetuned-bert" and have a local clone in the
            # `my-finetuned-bert` folder.
            {object}.push_to_hub("my-finetuned-bert")

            # Push the {object} to your namespace with the name "my-finetuned-bert" with no local clone.
            {object}.push_to_hub("my-finetuned-bert", use_temp_dir=True)

            # Push the {object} to an organization with the name "my-finetuned-bert" and have a local clone in the
            # `my-finetuned-bert` folder.
            {object}.push_to_hub("my-finetuned-bert", organization="huggingface")

            # Make a change to an existing repo that has been cloned locally in `my-finetuned-bert`.
            {object}.push_to_hub("my-finetuned-bert", repo_url="https://huggingface.co/sgugger/my-finetuned-bert")
        """
        ...
    


