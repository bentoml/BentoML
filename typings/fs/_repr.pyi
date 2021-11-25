

import typing
from typing import Text, Tuple

"""Tools to generate __repr__ strings.
"""
if typing.TYPE_CHECKING: ...

def make_repr(class_name: Text, *args: object, **kwargs: Tuple[object, object]) -> Text:
    """Generate a repr string.

    Positional arguments should be the positional arguments used to
    construct the class. Keyword arguments should consist of tuples of
    the attribute value and default. If the value is the default, then
    it won't be rendered in the output.

    Example:
        >>> class MyClass(object):
        ...     def __init__(self, name=None):
        ...         self.name = name
        ...     def __repr__(self):
        ...         return make_repr('MyClass', 'foo', name=(self.name, None))
        >>> MyClass('Will')
        MyClass('foo', name='Will')
        >>> MyClass(None)
        MyClass('foo')

    """
    ...
