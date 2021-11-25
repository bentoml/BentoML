

from pathlib import Path
from typing import Any, Dict, Optional, Union

REGEX_YAML_BLOCK = ...
def metadata_load(local_path: Union[str, Path]) -> Optional[Dict]:
    ...

def metadata_save(local_path: Union[str, Path], data: Dict) -> None:
    """
    Save the metadata dict in the upper YAML part
    Trying to preserve newlines as in the existing file.
    Docs about open() with newline="" parameter:
    https://docs.python.org/3/library/functions.html?highlight=open#open
    Does not work with "^M" linebreaks, which are replaced by \n
    """
    ...

def metadata_eval_result(model_pretty_name: str, task_pretty_name: str, task_id: str, metrics_pretty_name: str, metrics_id: str, metrics_value: Any, dataset_pretty_name: str, dataset_id: str) -> Dict:
    ...

