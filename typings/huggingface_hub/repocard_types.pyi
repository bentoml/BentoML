

from dataclasses import dataclass
from typing import Any, List, Optional

from typing_extensions import TypeAlias

ModelIndexSet: TypeAlias = ...
@dataclass
class ModelIndex:
    name: str
    results: List[SingleResult]
    ...


@dataclass
class SingleMetric:
    type: str
    value: Any
    args: Any = ...
    name: Optional[str] = ...


@dataclass
class SingleResultTask:
    type: str
    name: Optional[str] = ...


@dataclass
class SingleResultDataset:
    """
    This will switch to required at some point.
    in any case, we need them to link to PWC
    """
    name: str
    type: str
    args: Any = ...


@dataclass
class SingleResult:
    metrics: List[SingleMetric]
    task: SingleResultTask
    dataset: Optional[SingleResultDataset]
    ...


