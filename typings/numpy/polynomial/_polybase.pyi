import abc
from typing import Any, ClassVar, List

__all__: List[str]

class ABCPolyBase(abc.ABC):
    __hash__: ClassVar[None]
    __array_ufunc__: ClassVar[None]
    maxpower: ClassVar[int]
    coef: Any
    @property
    @abc.abstractmethod
    def domain(self): ...
    @property
    @abc.abstractmethod
    def window(self): ...
    @property
    @abc.abstractmethod
    def basis_name(self): ...
    def has_samecoef(self, other): ...
    def has_samedomain(self, other): ...
    def has_samewindow(self, other): ...
    def has_sametype(self, other): ...
    def __init__(self, coef, domain=..., window=...) -> None: ...
    def __format__(self, fmt_str): ...
    def __call__(self, arg): ...
    def __iter__(self): ...
    def __len__(self): ...
    def __neg__(self): ...
    def __pos__(self): ...
    def __add__(self, other): ...
    def __sub__(self, other): ...
    def __mul__(self, other): ...
    def __truediv__(self, other): ...
    def __floordiv__(self, other): ...
    def __mod__(self, other): ...
    def __divmod__(self, other): ...
    def __pow__(self, other): ...
    def __radd__(self, other): ...
    def __rsub__(self, other): ...
    def __rmul__(self, other): ...
    def __rdiv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __rfloordiv__(self, other): ...
    def __rmod__(self, other): ...
    def __rdivmod__(self, other): ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def copy(self): ...
    def degree(self): ...
    def cutdeg(self, deg): ...
    def trim(self, tol=...): ...
    def truncate(self, size): ...
    def convert(self, domain=..., kind=..., window=...): ...
    def mapparms(self): ...
    def integ(self, m=..., k=..., lbnd=...): ...
    def deriv(self, m=...): ...
    def roots(self): ...
    def linspace(self, n=..., domain=...): ...
    @classmethod
    def fit(cls, x, y, deg, domain=..., rcond=..., full=..., w=..., window=...): ...
    @classmethod
    def fromroots(cls, roots, domain=..., window=...): ...
    @classmethod
    def identity(cls, domain=..., window=...): ...
    @classmethod
    def basis(cls, deg, domain=..., window=...): ...
    @classmethod
    def cast(cls, series, domain=..., window=...): ...
