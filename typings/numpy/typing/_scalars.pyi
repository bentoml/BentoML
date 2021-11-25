from typing import Any, Tuple, Union

import numpy as np

_CharLike_co = Union[str, bytes]
_BoolLike_co = Union[bool, np.bool_]
_UIntLike_co = Union[_BoolLike_co, np.unsignedinteger]
_IntLike_co = Union[_BoolLike_co, int, np.integer]
_FloatLike_co = Union[_IntLike_co, float, np.floating]
_ComplexLike_co = Union[_FloatLike_co, complex, np.complexfloating]
_TD64Like_co = Union[_IntLike_co, np.timedelta64]
_NumberLike_co = Union[int, float, complex, np.number, np.bool_]
_ScalarLike_co = (Union[int, float, complex, str, bytes, np.generic],)
_VoidLike_co = Union[Tuple[Any, ...], np.void]
