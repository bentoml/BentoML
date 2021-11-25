import sys
from typing import Dict, List, Type, TypedDict, Union

from numpy import complexfloating, floating, generic, signedinteger, unsignedinteger

if sys.version_info >= (3, 8): ...
else: ...

class _SCTypes(TypedDict):
    int: List[Type[signedinteger]]
    uint: List[Type[unsignedinteger]]
    float: List[Type[floating]]
    complex: List[Type[complexfloating]]
    others: List[type]
    ...

sctypeDict: Dict[Union[int, str], Type[generic]]
sctypes: _SCTypes
