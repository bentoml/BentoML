import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Generic
from typing import Mapping
from typing import TypeVar
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING

import attrs
import click
import cerberus.errors

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

    ReleaseTagInfo = Dict[str, Dict[str, Union[List[str], str]]]

    class ClickFunctionWrapper(Generic[P, T]):
        __name__: str
        __click_params__: List[click.Option]

        def __call__(*args: P.args, **kwargs: P.kwargs) -> Callable[P, T]:
            ...

    WrappedCLI = Callable[Concatenate[P], ClickFunctionWrapper[Any, Any]]

    ValidateType = Union[str, List[str]]

    class CerberusValidator:
        root_document: Dict[str, Any]
        document: Dict[str, Any]
        errors: cerberus.errors.ErrorList

        def _error(self, field: str, msg: str):
            ...

        def validate(
            self,
            document: Mapping[str, Any],
            schema: Optional[str] = None,
            update: bool = False,
            normalize: bool = True,
        ):
            ...

        @classmethod
        def clear_caches(cls):
            ...

        def normalized(
            self,
            document: Mapping[str, Any],
            schema: Optional[str] = None,
            always_return_document: bool = False,
        ) -> Dict[str, Any]:
            ...

    class AttrsCls:
        __attrs_attrs__: attrs.Attribute

    DictStrAttrs = Dict[str, AttrsCls]
