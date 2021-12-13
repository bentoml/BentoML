import sys, six

_PY = sys.version_info
from typing import overload

try:
    from typing import Text
except ImportError:
    Text = six.text_type
