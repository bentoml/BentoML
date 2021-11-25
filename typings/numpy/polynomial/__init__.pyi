from typing import List

from numpy.polynomial import chebyshev as chebyshev
from numpy.polynomial import hermite as hermite
from numpy.polynomial import hermite_e as hermite_e
from numpy.polynomial import laguerre as laguerre
from numpy.polynomial import legendre as legendre
from numpy.polynomial import polynomial as polynomial
from numpy.polynomial.chebyshev import Chebyshev as Chebyshev
from numpy.polynomial.hermite import Hermite as Hermite
from numpy.polynomial.hermite_e import HermiteE as HermiteE
from numpy.polynomial.laguerre import Laguerre as Laguerre
from numpy.polynomial.legendre import Legendre as Legendre
from numpy.polynomial.polynomial import Polynomial as Polynomial

__all__: List[str]

def set_default_printstyle(style): ...
