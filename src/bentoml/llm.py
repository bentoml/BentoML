# fmt: off
from __future__ import annotations
import sys, typing as t
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
if t.TYPE_CHECKING:from openllm import LLMRunner as LLMRunner,LLMRunnable as LLMRunnable,Runner as Runner,LLM as LLM,llm_runner_class as llm_runner_class,llm_runnable_class as llm_runnable_class,LLMEmbeddings as LLMEmbeddings
@overload
def __getattr__(item:Literal['LLMRunner'])->type[LLMRunner]:...
@overload
def __getattr__(item:Literal['LLMRunnable'])->type[LLMRunnable]:...
@overload
def __getattr__(item:Literal['Runner'])->t.Callable[...,t.Any]:...
@overload
def __getattr__(item:Literal['LLM'])->type[LLM]:...
@overload
def __getattr__(item:Literal['llm_runner_class'])->t.Callable[...,t.Any]:...
@overload
def __getattr__(item:Literal['llm_runnable_class'])->t.Callable[...,t.Any]:...
@overload
def __getattr__(item:Literal['LLMEmbeddings'])->type[LLMEmbeddings]:...
def __getattr__(item:t.Any)->t.Any:return getattr(openllm,item)
# update_llm_imports.py: end
