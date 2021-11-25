from . import _inspect, py3k
from ._inspect import formatargspec, getargspec
from .py3k import *

"""
Compatibility module.

This module contains duplicated code from Python itself or 3rd party
extensions, which may be included for the following reasons:

  * compatibility
  * we may only need a small subset of the copied library/module

"""
__all__ = []
