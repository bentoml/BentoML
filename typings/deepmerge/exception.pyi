from typing import Any, Dict, Tuple

class DeepMergeException(Exception): ...
class StrategyNotFound(DeepMergeException): ...

class InvalidMerge(DeepMergeException):
    def __init__(
        self,
        strategy_list_name: str,
        merge_args: Tuple[Any],
        merge_kwargs: Dict[str, Any],
    ) -> None: ...
