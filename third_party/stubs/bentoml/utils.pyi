from ._internal.configuration.containers import BentoMLContainer as BentoMLContainer
from typing import Any, Dict, List, Optional

SUPPORTED_PYTHON_VERSION: List
SUPPORTED_BASE_DISTROS: List
SUPPORTED_GPU_DISTROS: List
SUPPORTED_RELEASES_COMBINATION: Dict[str, List[str]]
SEMVER_REGEX: Any
logger: Any
BACKWARD_COMPATIBILITY_WARNING: str

def get_suffix(gpu: bool) -> str: ...

class ImageProvider:
    def __init__(self, distros: str, gpu: bool = ..., python_version: Optional[str] = ..., bentoml_version: Optional[str] = ...) -> None: ...
    def __new__(cls, *args, **kwargs): ...
