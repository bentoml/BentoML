from __future__ import unicode_literals
from . import patterns
from ._meta import __author__, __copyright__, __credits__, __license__, __version__
from .pathspec import PathSpec
from .pattern import Pattern, RegexPattern
from .patterns.gitwildmatch import GitIgnorePattern
from .util import RecursionError, iter_tree, lookup_pattern, match_files

__all__ = ["PathSpec"]
