

from pathlib import Path
from typing import Dict, Optional, Union

REPO_ID_SEPARATOR = ...
def snapshot_download(repo_id: str, revision: Optional[str] = ..., cache_dir: Union[str, Path, None] = ..., library_name: Optional[str] = ..., library_version: Optional[str] = ..., user_agent: Union[Dict, str, None] = ..., use_auth_token: Union[bool, str, None] = ...) -> str:
    """
    Downloads a whole snapshot of a repo's files at the specified revision.
    This is useful when you want all files from a repo, because you don't know
    which ones you will need a priori.
    All files are nested inside a folder in order to keep their actual filename
    relative to that folder.

    An alternative would be to just clone a repo but this would require that
    the user always has git and git-lfs installed, and properly configured.

    Note: at some point maybe this format of storage should actually replace
    the flat storage structure we've used so far (initially from allennlp
    if I remember correctly).

    Return:
        Local folder path (string) of repo snapshot
    """
    ...

