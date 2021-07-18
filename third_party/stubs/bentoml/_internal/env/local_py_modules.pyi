from bentoml.exceptions import BentoMLException as BentoMLException
from bentoml.saved_bundle.pip_pkg import get_all_pip_installed_modules as get_all_pip_installed_modules
from typing import Any, List
from unittest.mock import patch as patch

logger: Any

def copy_local_py_modules(target_module, destination): ...
def copy_zip_import_archives(target_path: str, target_module: str, inferred_zipimports: List[str], zipimports: List[str]): ...
