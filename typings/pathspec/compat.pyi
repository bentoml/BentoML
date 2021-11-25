

import sys
from collections.abc import Iterable

"""
This module provides compatibility between Python 2 and 3. Hardly
anything is used by this project to constitute including `six`_.

.. _`six`: http://pythonhosted.org/six
"""
if sys.version_info[0] < 3: ...
else:
    unicode = str
    string_types = ...
    def iterkeys(mapping): ...

CollectionType = Collection
IterableType = Iterable
