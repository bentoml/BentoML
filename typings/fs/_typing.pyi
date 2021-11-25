

"""
Typing objects missing from Python3.5.1

"""
import sys

import six

_PY = sys.version_info

from typing import overload  # type: ignore

try:
    from typing import Text  # type: ignore
except ImportError:  # pragma: no cover
    Text = six.text_type  # type: ignore
