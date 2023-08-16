# fmt: off
from __future__ import annotations

import sys
import typing as t

from .exceptions import MissingDependencyException

try: import openllm
except ImportError: raise MissingDependencyException("'openllm' is required to use with 'bentoml.llm'. Make sure to install it with 'pip install \"bentoml[llm]\"'") from None

if sys.version_info[:2] >= (3,11): from typing import overload
else: from typing_extensions import overload
if sys.version_info[:2] >= (3,8): from typing import Literal
else: from typing_extensions import Literal

# NOTE: The following are managed via ./tools/update_llm_imports.py
# update_llm_imports.py: start
def __dir__()->list[str]:return sorted(openllm._llm.__all__)
@overload
def __getattr__(item: Literal['openllm.LLMRunner'])->t.Any:...
@overload
def __getattr__(item: Literal['openllm.LLMRunnable'])->t.Any:...
@overload
def __getattr__(item: Literal['openllm.Runner'])->t.Any:...
@overload
def __getattr__(item: Literal['openllm.LLM'])->t.Any:...
@overload
def __getattr__(item: Literal['openllm.llm_runner_class'])->t.Any:...
@overload
def __getattr__(item: Literal['openllm.llm_runnable_class'])->t.Any:...
@overload
def __getattr__(item: Literal['openllm.LLMEmbeddings'])->t.Any:...
def __getattr__(item: t.Any)->t.Any:return getattr(openllm, item)
# update_llm_imports.py: end
# fmt: on
