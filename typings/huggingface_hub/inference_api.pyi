from typing import Dict, List, Optional, Union

logger = ...
ENDPOINT = ...
ALL_TASKS = ...

class InferenceApi:
    def __init__(
        self,
        repo_id: str,
        task: Optional[str] = ...,
        token: Optional[str] = ...,
        gpu: Optional[bool] = ...,
    ) -> None: ...
    def __repr__(self): ...
    def __call__(
        self,
        inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = ...,
        params: Optional[Dict] = ...,
        data: Optional[bytes] = ...,
    ): ...
