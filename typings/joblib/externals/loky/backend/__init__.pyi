import os
import sys
from multiprocessing import synchronize
from .context import get_context

if sys.version_info > (3, 4): ...
__all__ = ["get_context"]
