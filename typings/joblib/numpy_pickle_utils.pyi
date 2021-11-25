import pickle

"""Utilities for fast persistence of big data, with optional compression."""
Unpickler = pickle._Unpickler
Pickler = pickle._Pickler
xrange = range
_IO_BUFFER_SIZE = ...
BUFFER_SIZE = 2 ** 18
