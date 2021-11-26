import sys
from collections.abc import Iterable

if sys.version_info[0] < 3: ...
else:
    unicode = str
    string_types = ...
    def iterkeys(mapping): ...

CollectionType = Collection
IterableType = Iterable
