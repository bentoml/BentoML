from __future__ import absolute_import

from .cloudpickle import *
from .cloudpickle_fast import CloudPickler, dump, dumps

Pickler = CloudPickler
__version__ = ...
