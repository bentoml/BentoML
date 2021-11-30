from typing import Any, List
from numpy._pytesttester import PytestTester
from numpy.linalg.linalg import cholesky as cholesky
from numpy.linalg.linalg import cond as cond
from numpy.linalg.linalg import det as det
from numpy.linalg.linalg import eig as eig
from numpy.linalg.linalg import eigh as eigh
from numpy.linalg.linalg import eigvals as eigvals
from numpy.linalg.linalg import eigvalsh as eigvalsh
from numpy.linalg.linalg import inv as inv
from numpy.linalg.linalg import lstsq as lstsq
from numpy.linalg.linalg import matrix_power as matrix_power
from numpy.linalg.linalg import matrix_rank as matrix_rank
from numpy.linalg.linalg import multi_dot as multi_dot
from numpy.linalg.linalg import norm as norm
from numpy.linalg.linalg import pinv as pinv
from numpy.linalg.linalg import qr as qr
from numpy.linalg.linalg import slogdet as slogdet
from numpy.linalg.linalg import solve as solve
from numpy.linalg.linalg import svd as svd
from numpy.linalg.linalg import tensorinv as tensorinv
from numpy.linalg.linalg import tensorsolve as tensorsolve

__all__: List[str]
__path__: List[str]
test: PytestTester

class LinAlgError(Exception): ...
