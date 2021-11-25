

import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

logger = ...
class CommandInProgress:
    def __init__(self, title: str, is_done_method: Callable, status_method: Callable, process: subprocess.Popen, post_method: Optional[Callable] = ...) -> None:
        ...
    
    @property
    def is_done(self) -> bool:
        """
        Whether the process is done.
        """
        ...
    
    @property
    def status(self) -> int:
        """
        The exit code/status of the current action. Will return `0` if the command has completed
        successfully, and a number between 1 and 255 if the process errored-out.

        Will return -1 if the command is still ongoing.
        """
        ...
    
    @property
    def failed(self) -> bool:
        """
        Whether the process errored-out.
        """
        ...
    
    @property
    def stderr(self) -> str:
        """
        The current output message on the standard error.
        """
        ...
    
    @property
    def stdout(self) -> str:
        """
        The current output message on the standard output.
        """
        ...
    
    def __repr__(self): # -> str:
        ...
    


def is_git_repo(folder: Union[str, Path]) -> bool:
    """
    Check if the folder is the root of a git repository
    """
    ...

def is_local_clone(folder: Union[str, Path], remote_url: str) -> bool:
    """
    Check if the folder is the a local clone of the remote_url
    """
    ...

def is_tracked_with_lfs(filename: Union[str, Path]) -> bool:
    """
    Check if the file passed is tracked with git-lfs.
    """
    ...

def is_git_ignored(filename: Union[str, Path]) -> bool:
    """
    Check if file is git-ignored. Supports nested .gitignore files.
    """
    ...

def files_to_be_staged(pattern: str, folder: Union[str, Path]) -> List[str]:
    ...

def is_tracked_upstream(folder: Union[str, Path]) -> bool:
    """
    Check if the current checked-out branch is tracked upstream.
    """
    ...

def commits_to_push(folder: Union[str, Path], upstream: Optional[str] = ...) -> int:
    """
    Check the number of commits that would be pushed upstream
    """
    ...

@contextmanager
def lfs_log_progress(): # -> Generator[None, None, None]:
    """
    This is a context manager that will log the Git LFS progress of cleaning, smudging, pulling and pushing.
    """
    ...

class Repository:
    """
    Helper class to wrap the git and git-lfs commands.

    The aim is to facilitate interacting with huggingface.co hosted model or dataset repos,
    though not a lot here (if any) is actually specific to huggingface.co.
    """
    command_queue: List[CommandInProgress]
    def __init__(self, local_dir: str, clone_from: Optional[str] = ..., repo_type: Optional[str] = ..., use_auth_token: Union[bool, str] = ..., git_user: Optional[str] = ..., git_email: Optional[str] = ..., revision: Optional[str] = ..., private: bool = ...) -> None:
        """
        Instantiate a local clone of a git repo.

        If specifying a `clone_from`:
        will clone an existing remote repository, for instance one
        that was previously created using ``HfApi().create_repo(name=repo_name)``.
        ``Repository`` uses the local git credentials by default, but if required, the ``huggingface_token``
        as well as the git ``user`` and the ``email`` can be explicitly specified.
        If `clone_from` is used, and the repository is being instantiated into a non-empty directory,
        e.g. a directory with your trained model files, it will automatically merge them.

        Args:
            local_dir (``str``):
                path (e.g. ``'my_trained_model/'``) to the local directory, where the ``Repository`` will be initalized.
            clone_from (``str``, `optional`):
                repository url (e.g. ``'https://huggingface.co/philschmid/playground-tests'``).
            repo_type (``str``, `optional`):
                To set when creating a repo: et to "dataset" or "space" if creating a dataset or space, default is model.
            use_auth_token (``str`` or ``bool``, `optional`, defaults ``None``):
                huggingface_token can be extract from ``HfApi().login(username, password)`` and is used to authenticate against the hub
                (useful from Google Colab for instance).
            git_user (``str``, `optional`):
                will override the ``git config user.name`` for committing and pushing files to the hub.
            git_email (``str``, `optional`):
                will override the ``git config user.email`` for committing and pushing files to the hub.
            revision (``str``, `optional`):
                Revision to checkout after initializing the repository. If the revision doesn't exist, a
                branch will be created with that revision name from the default branch's current HEAD.
            private (``bool``, `optional`):
                whether the repository is private or not.
        """
        ...
    
    @property
    def current_branch(self): # -> str:
        """
        Returns the current checked out branch.
        """
        ...
    
    def check_git_versions(self): # -> None:
        """
        print git and git-lfs versions, raises if they aren't installed.
        """
        ...
    
    def clone_from(self, repo_url: str, use_auth_token: Union[bool, str, None] = ...): # -> None:
        """
        Clone from a remote. If the folder already exists, will try to clone the repository within it.

        If this folder is a git repository with linked history, will try to update the repository.
        """
        ...
    
    def git_config_username_and_email(self, git_user: Optional[str] = ..., git_email: Optional[str] = ...): # -> None:
        """
        sets git user name and email (only in the current repo)
        """
        ...
    
    def git_credential_helper_store(self): # -> None:
        """
        sets the git credential helper to `store`
        """
        ...
    
    def git_head_hash(self) -> str:
        """
        Get commit sha on top of HEAD.
        """
        ...
    
    def git_remote_url(self) -> str:
        """
        Get URL to origin remote.
        """
        ...
    
    def git_head_commit_url(self) -> str:
        """
        Get URL to last commit on HEAD
        We assume it's been pushed, and the url scheme is
        the same one as for GitHub or HuggingFace.
        """
        ...
    
    def list_deleted_files(self) -> List[str]:
        """
        Returns a list of the files that are deleted in the working directory or index.
        """
        ...
    
    def lfs_track(self, patterns: Union[str, List[str]], filename: Optional[bool] = ...): # -> None:
        """
        Tell git-lfs to track those files.

        Setting the `filename` argument to `True` will treat the arguments as literal filenames,
        not as patterns. Any special glob characters in the filename will be escaped when
        writing to the .gitattributes file.
        """
        ...
    
    def lfs_untrack(self, patterns: Union[str, List[str]]): # -> None:
        """
        Tell git-lfs to untrack those files.
        """
        ...
    
    def lfs_enable_largefiles(self): # -> None:
        """
        HF-specific. This enables upload support of files >5GB.
        """
        ...
    
    def auto_track_large_files(self, pattern: Optional[str] = ...) -> List[str]:
        """
        Automatically track large files with git-lfs
        """
        ...
    
    def lfs_prune(self, recent=...): # -> None:
        """
        git lfs prune
        """
        ...
    
    def git_pull(self, rebase: Optional[bool] = ...): # -> None:
        """
        git pull
        """
        ...
    
    def git_add(self, pattern: Optional[str] = ..., auto_lfs_track: Optional[bool] = ...): # -> None:
        """
        git add

        Setting the `auto_lfs_track` parameter to `True` will automatically track files that are larger
        than 10MB with `git-lfs`.
        """
        ...
    
    def git_commit(self, commit_message: str = ...): # -> None:
        """
        git commit
        """
        ...
    
    def git_push(self, upstream: Optional[str] = ..., blocking: Optional[bool] = ..., auto_lfs_prune: Optional[bool] = ...) -> Union[str, Tuple[str, CommandInProgress]]:
        """
        git push

        If used without setting `blocking`, will return url to commit on remote repo.
        If used with `blocking=True`, will return a tuple containing the url to commit
        and the command object to follow for information about the process.

        Args:
            upstream (`str`, `optional`):
                Upstream to which this should push. If not specified, will push
                to the lastly defined upstream or to the default one (`origin main`).
            blocking (`bool`, defaults to `True`):
                Whether the function should return only when the push has finished.
                Setting this to `False` will return an `CommandInProgress` object
                which has an `is_done` property. This property will be set to
                `True` when the push is finished.
            auto_lfs_prune (`bool`, defaults to `False`):
                Whether to automatically prune files once they have been pushed to the remote.
        """
        ...
    
    def git_checkout(self, revision: str, create_branch_ok: Optional[bool] = ...): # -> None:
        """
        git checkout a given revision

        Specifying `create_branch_ok` to `True` will create the branch to the given revision if that revision doesn't exist.
        """
        ...
    
    def tag_exists(self, tag_name: str, remote: Optional[str] = ...) -> bool:
        """
        Check if a tag exists or not
        """
        ...
    
    def delete_tag(self, tag_name: str, remote: Optional[str] = ...) -> bool:
        """
        Delete a tag, both local and remote, if it exists

        Return True if deleted.  Returns False if the tag didn't exist
        If remote is None, will just be updated locally
        """
        ...
    
    def add_tag(self, tag_name: str, message: str = ..., remote: Optional[str] = ...): # -> None:
        """
        Add a tag at the current head and push it

        If remote is None, will just be updated locally

        If no message is provided, the tag will be lightweight.
        if a message is provided, the tag will be annotated.
        """
        ...
    
    def is_repo_clean(self) -> bool:
        """
        Return whether or not the git status is clean or not
        """
        ...
    
    def push_to_hub(self, commit_message: Optional[str] = ..., blocking: Optional[bool] = ..., clean_ok: Optional[bool] = ..., auto_lfs_prune: Optional[bool] = ...) -> Optional[str]:
        """
        Helper to add, commit, and push files to remote repository on the HuggingFace Hub.
        Will automatically track large files (>10MB).

        Args:
            commit_message (`str`):
                Message to use for the commit.
            blocking (`bool`, `optional`, defaults to `True`):
                Whether the function should return only when the `git push` has finished.
            clean_ok (`bool`, `optional`, defaults to `True`):
                If True, this function will return None if the repo is untouched.
                Default behavior is to fail because the git command fails.
            auto_lfs_prune (`bool`, defaults to `False`):
                Whether to automatically prune files once they have been pushed to the remote.
        """
        ...
    
    @contextmanager
    def commit(self, commit_message: str, branch: Optional[str] = ..., track_large_files: Optional[bool] = ..., blocking: Optional[bool] = ..., auto_lfs_prune: Optional[bool] = ...): # -> Generator[Self@Repository, None, None]:
        """
        Context manager utility to handle committing to a repository. This automatically tracks large files (>10Mb)
        with git-lfs. Set the `track_large_files` argument to `False` if you wish to ignore that behavior.

        Args:
            commit_message (`str`):
                Message to use for the commit.
            branch (`str`, `optional`):
                The branch on which the commit will appear. This branch will be checked-out before any operation.
            track_large_files (`bool`, `optional`, defaults to `True`):
                Whether to automatically track large files or not. Will do so by default.
            blocking (`bool`, `optional`, defaults to `True`):
                Whether the function should return only when the `git push` has finished.
            auto_lfs_prune (`bool`, defaults to `True`):
                Whether to automatically prune files once they have been pushed to the remote.

        Examples:

            >>> with Repository("text-files", clone_from="<user>/text-files", use_auth_token=True).commit("My first file :)"):
            ...     with open("file.txt", "w+") as f:
            ...         f.write(json.dumps({"hey": 8}))

            >>> import torch
            >>> model = torch.nn.Transformer()
            >>> with Repository("torch-model", clone_from="<user>/torch-model", use_auth_token=True).commit("My cool model :)"):
            ...     torch.save(model.state_dict(), "model.pt")

        """
        ...
    
    def repocard_metadata_load(self) -> Optional[Dict]:
        ...
    
    def repocard_metadata_save(self, data: Dict) -> None:
        ...
    
    @property
    def commands_failed(self): # -> list[CommandInProgress]:
        ...
    
    @property
    def commands_in_progress(self): # -> list[CommandInProgress]:
        ...
    
    def wait_for_commands(self): # -> None:
        ...
    


