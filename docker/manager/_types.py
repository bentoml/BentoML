import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Generic
from typing import TypeVar
from typing import Callable
from typing import TYPE_CHECKING

import click

if sys.version_info[:2] >= (3, 10):
    from typing import ParamSpec
    from typing import Concatenate
else:
    from typing_extensions import ParamSpec
    from typing_extensions import Concatenate

if TYPE_CHECKING:
    P = ParamSpec("P")
    T = TypeVar("T")

    StrDict = Dict[str, str]
    StrTuple = Tuple[str, str]
    StrList = List[str]

    GenericDict = Dict[str, Any]
    GenericList = List[Any]
    GenericFunc = Callable[P, Any]
    GenericNestedDict = Dict[str, Dict[str, Any]]
    DoubleNestedDict = Dict[str, Dict[str, T]]

    ReleaseTagInfo = Tuple[str, Dict[str, StrDict]]

    class ClickFunctionWrapper(Generic[P, T]):
        __name__: str
        __click_params__: List[click.Option]

        params: List[click.Parameter]

        def __call__(*args: P.args, **kwargs: P.kwargs) -> Callable[P, T]:
            ...

    WrappedCLI = Callable[Concatenate[P], ClickFunctionWrapper[Any, Any]]

    ValidateType = Union[str, List[str]]
