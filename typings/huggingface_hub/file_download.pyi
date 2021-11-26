from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple, Union

logger = ...
_PY_VERSION: str = ...
if packaging.version.Version(_PY_VERSION) < packaging.version.Version("3.8.0"): ...
else: ...
_torch_version = ...
_torch_available = ...
_tf_version = ...
_tf_available = ...
_tf_candidates = ...

def is_torch_available(): ...
def is_tf_available(): ...
def hf_hub_url(
    repo_id: str,
    filename: str,
    subfolder: Optional[str] = ...,
    repo_type: Optional[str] = ...,
    revision: Optional[str] = ...,
) -> str: ...
def url_to_filename(url: str, etag: Optional[str] = ...) -> str: ...
def filename_to_url(filename, cache_dir=...) -> Tuple[str, str]: ...
def http_user_agent(
    library_name: Optional[str] = ...,
    library_version: Optional[str] = ...,
    user_agent: Union[Dict, str, None] = ...,
) -> str: ...

class OfflineModeIsEnabled(ConnectionError): ...

def http_get(
    url: str,
    temp_file: BinaryIO,
    proxies=...,
    resume_size=...,
    headers: Optional[Dict[str, str]] = ...,
    timeout=...,
    max_retries=...,
): ...
def cached_download(
    url: str,
    library_name: Optional[str] = ...,
    library_version: Optional[str] = ...,
    cache_dir: Union[str, Path, None] = ...,
    user_agent: Union[Dict, str, None] = ...,
    force_download=...,
    force_filename: Optional[str] = ...,
    proxies=...,
    etag_timeout=...,
    resume_download=...,
    use_auth_token: Union[bool, str, None] = ...,
    local_files_only=...,
) -> Optional[str]: ...
def hf_hub_download(
    repo_id: str,
    filename: str,
    subfolder: Optional[str] = ...,
    repo_type: Optional[str] = ...,
    revision: Optional[str] = ...,
    library_name: Optional[str] = ...,
    library_version: Optional[str] = ...,
    cache_dir: Union[str, Path, None] = ...,
    user_agent: Union[Dict, str, None] = ...,
    force_download=...,
    force_filename: Optional[str] = ...,
    proxies=...,
    etag_timeout=...,
    resume_download=...,
    use_auth_token: Union[bool, str, None] = ...,
    local_files_only=...,
): ...
