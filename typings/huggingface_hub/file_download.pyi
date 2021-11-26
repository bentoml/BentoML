from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple, Union

logger = ...
_PY_VERSION: str = ...
if packaging.version.Version(_PY_VERSION) < packaging.version.Version("3.8.0"):
    ...
else:
    ...
_torch_version = ...
_torch_available = ...
_tf_version = ...
_tf_available = ...
_tf_candidates = ...
def is_torch_available(): # -> bool:
    ...

def is_tf_available(): # -> bool:
    ...

def hf_hub_url(repo_id: str, filename: str, subfolder: Optional[str] = ..., repo_type: Optional[str] = ..., revision: Optional[str] = ...) -> str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co-hosted url, redirecting
    to Cloudfront (a Content Delivery Network, or CDN) for large files (more than a few MBs).

    Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
    bandwidth costs).

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we implement a git-based versioning system on huggingface.co, which means that we store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object's ETag is:
    its git-sha1 if stored in git, or its sha256 if stored in git-lfs.
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

def filename_to_url(filename, cache_dir=...) -> Tuple[str, str]:
    """
    Return the url and etag (which may be ``None``) stored for `filename`. Raise ``EnvironmentError`` if `filename` or
    its stored metadata do not exist.
    """
    ...

def http_user_agent(library_name: Optional[str] = ..., library_version: Optional[str] = ..., user_agent: Union[Dict, str, None] = ...) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ...

class OfflineModeIsEnabled(ConnectionError):
    ...


def http_get(url: str, temp_file: BinaryIO, proxies=..., resume_size=..., headers: Optional[Dict[str, str]] = ..., timeout=..., max_retries=...): # -> None:
    """
    Donwload remote file. Do not gobble up errors.
    """
    ...

def cached_download(url: str, library_name: Optional[str] = ..., library_version: Optional[str] = ..., cache_dir: Union[str, Path, None] = ..., user_agent: Union[Dict, str, None] = ..., force_download=..., force_filename: Optional[str] = ..., proxies=..., etag_timeout=..., resume_download=..., use_auth_token: Union[bool, str, None] = ..., local_files_only=...) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    ...

def hf_hub_download(repo_id: str, filename: str, subfolder: Optional[str] = ..., repo_type: Optional[str] = ..., revision: Optional[str] = ..., library_name: Optional[str] = ..., library_version: Optional[str] = ..., cache_dir: Union[str, Path, None] = ..., user_agent: Union[Dict, str, None] = ..., force_download=..., force_filename: Optional[str] = ..., proxies=..., etag_timeout=..., resume_download=..., use_auth_token: Union[bool, str, None] = ..., local_files_only=...): # -> str | None:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co file distributed through
    Cloudfront (a Content Delivery Network, or CDN) for large files (more than a few MBs).

    The file is cached locally: look for the corresponding file in the local cache. If it's not there,
    download it. Then return the path to the cached file.

    Cloudfront is replicated over the globe so downloads are way faster for the end user.

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we implement a git-based versioning system on huggingface.co, which means that we store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object's ETag is:
    its git-sha1 if stored in git, or its sha256 if stored in git-lfs.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    ...

