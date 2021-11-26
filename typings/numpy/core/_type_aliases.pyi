from typing import Dict, List, Type, TypedDict, Union
from numpy import complexfloating, floating, generic, signedinteger, unsignedinteger

class _SCTypes(TypedDict):
    int: List[Type[signedinteger]]
    uint: List[Type[unsignedinteger]]
    float: List[Type[floating]]
    complex: List[Type[complexfloating]]
    others: List[type]

sctypeDict: Dict[Union[int, str], Type[generic]]
sctypes: _SCTypes
