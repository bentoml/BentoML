
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from . import Config

if TYPE_CHECKING:
    ...
def load_config_dict_from_file(filepath: Path) -> Optional[Dict[str, Union[str, List[str]]]]:
    """Load pytest configuration from the given file path, if supported.

    Return None if the file does not contain valid pytest configuration.
    """
    ...

def locate_config(args: Iterable[Path]) -> Tuple[Optional[Path], Optional[Path], Dict[str, Union[str, List[str]]]],:
    """Search in the list of arguments for a valid ini-file for pytest,
    and return a tuple of (rootdir, inifile, cfg-dict)."""
    ...

def get_common_ancestor(paths: Iterable[Path]) -> Path:
    ...

def get_dirs_from_args(args: Iterable[str]) -> List[Path]:
    ...

CFG_PYTEST_SECTION = ...
def determine_setup(inifile: Optional[str], args: Sequence[str], rootdir_cmd_arg: Optional[str] = ..., config: Optional[Config] = ...) -> Tuple[Path, Optional[Path], Dict[str, Union[str, List[str]]]]:
    ...

