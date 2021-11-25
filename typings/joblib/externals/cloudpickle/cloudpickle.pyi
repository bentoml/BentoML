import sys

"""
This class is defined to override standard pickle functionality

The goals of it follow:
-Serialize lambdas and nested functions to compiled byte code
-Deal with main module correctly
-Deal with other non-serializable objects

It does not include an unpickler, as standard python unpickling suffices.

This module was extracted from the `cloud` package, developed by `PiCloud, Inc.
<https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.

Copyright (c) 2012, Regents of the University of California.
Copyright (c) 2009 `PiCloud, Inc. <https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the University of California, Berkeley nor the
      names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
if sys.version_info >= (3, 5, 3): ...
else: ...
if sys.version_info >= (3, 8): ...
else: ...
DEFAULT_PROTOCOL = ...
_PICKLE_BY_VALUE_MODULES = ...
_DYNAMIC_CLASS_TRACKER_BY_CLASS = ...
_DYNAMIC_CLASS_TRACKER_BY_ID = ...
_DYNAMIC_CLASS_TRACKER_LOCK = ...
PYPY = ...
builtin_code_type = ...
if PYPY:
    builtin_code_type = ...
_extract_code_globals_cache = ...

def register_pickle_by_value(module):  # -> None:
    """Register a module to make it functions and classes picklable by value.

    By default, functions and classes that are attributes of an importable
    module are to be pickled by reference, that is relying on re-importing
    the attribute from the module at load time.

    If `register_pickle_by_value(module)` is called, all its functions and
    classes are subsequently to be pickled by value, meaning that they can
    be loaded in Python processes where the module is not importable.

    This is especially useful when developing a module in a distributed
    execution environment: restarting the client Python process with the new
    source code is enough: there is no need to re-install the new version
    of the module on all the worker nodes nor to restart the workers.

    Note: this feature is considered experimental. See the cloudpickle
    README.md file for more details and limitations.
    """
    ...

def unregister_pickle_by_value(module):  # -> None:
    """Unregister that the input module should be pickled by value."""
    ...

def list_registry_pickle_by_value(): ...
def cell_set(cell, value):  # -> None:
    """Set the value of a closure cell.

    The point of this function is to set the cell_contents attribute of a cell
    after its creation. This operation is necessary in case the cell contains a
    reference to the function the cell belongs to, as when calling the
    function's constructor
    ``f = types.FunctionType(code, globals, name, argdefs, closure)``,
    closure will not be able to contain the yet-to-be-created f.

    In Python3.7, cell_contents is writeable, so setting the contents of a cell
    can be done simply using
    >>> cell.cell_contents = value

    In earlier Python3 versions, the cell_contents attribute of a cell is read
    only, but this limitation can be worked around by leveraging the Python 3
    ``nonlocal`` keyword.

    In Python2 however, this attribute is read only, and there is no
    ``nonlocal`` keyword. For this reason, we need to come up with more
    complicated hacks to set this attribute.

    The chosen approach is to create a function with a STORE_DEREF opcode,
    which sets the content of a closure variable. Typically:

    >>> def inner(value):
    ...     lambda: cell  # the lambda makes cell a closure
    ...     cell = value  # cell is a closure, so this triggers a STORE_DEREF

    (Note that in Python2, A STORE_DEREF can never be triggered from an inner
    function. The function g for example here
    >>> def f(var):
    ...     def g():
    ...         var += 1
    ...     return g

    will not modify the closure variable ``var```inplace, but instead try to
    load a local variable var and increment it. As g does not assign the local
    variable ``var`` any initial value, calling f(1)() will fail at runtime.)

    Our objective is to set the value of a given cell ``cell``. So we need to
    somewhat reference our ``cell`` object into the ``inner`` function so that
    this object (and not the smoke cell of the lambda function) gets affected
    by the STORE_DEREF operation.

    In inner, ``cell`` is referenced as a cell variable (an enclosing variable
    that is referenced by the inner function). If we create a new function
    cell_set with the exact same code as ``inner``, but with ``cell`` marked as
    a free variable instead, the STORE_DEREF will be applied on its closure -
    ``cell``, which we can specify explicitly during construction! The new
    cell_set variable thus actually sets the contents of a specified cell!

    Note: we do not make use of the ``nonlocal`` keyword to set the contents of
    a cell in early python3 versions to limit possible syntax errors in case
    test and checker libraries decide to parse the whole file.
    """
    ...

if sys.version_info[:2] < (3, 7):
    _cell_set_template_code = ...
STORE_GLOBAL = ...
DELETE_GLOBAL = ...
LOAD_GLOBAL = ...
GLOBAL_OPS = ...
HAVE_ARGUMENT = ...
EXTENDED_ARG = ...
_BUILTIN_TYPE_NAMES = ...
if sys.version_info[:2] < (3, 7): ...
else:
    _is_parametrized_type_hint = ...
    _create_parametrized_type_hint = ...

def parametrized_type_hint_getinitargs(obj): ...
def is_tornado_coroutine(func):  # -> Literal[False]:
    """
    Return whether *func* is a Tornado coroutine function.
    Running coroutines are not supported.
    """
    ...

load = ...
loads = ...

def subimport(name): ...
def dynamic_subimport(name, vars): ...
def instance(cls):
    """Create a new instance of a class.

    Parameters
    ----------
    cls : type
        The class to create an instance of.

    Returns
    -------
    instance : cls
        A new instance of ``cls``.
    """
    ...

@instance
class _empty_cell_value:
    """sentinel for empty closures"""

    @classmethod
    def __reduce__(cls): ...
