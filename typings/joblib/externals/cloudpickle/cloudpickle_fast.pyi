from .cloudpickle import PYPY
from .compat import Pickler, pickle

"""
New, fast version of the CloudPickler.

This new CloudPickler class can now extend the fast C Pickler instead of the
previous Python implementation of the Pickler class. Because this functionality
is only available for Python versions 3.8+, a lot of backward-compatibility
code is also removed.

Note that the C Pickler subclassing API is CPython-specific. Therefore, some
guards present in cloudpickle.py that were written to handle PyPy specificities
are not present in cloudpickle_fast.py
"""
if pickle.HIGHEST_PROTOCOL >= 5 and notPYPY:
    def dump(obj, file, protocol=..., buffer_callback=...):  # -> None:
        """Serialize obj as bytes streamed into file

        protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
        pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
        speed between processes running the same Python version.

        Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
        compatibility with older versions of Python.
        """
        ...
    def dumps(obj, protocol=..., buffer_callback=...):  # -> bytes:
        """Serialize obj as a string of bytes allocated in memory

        protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
        pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
        speed between processes running the same Python version.

        Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
        compatibility with older versions of Python.
        """
        ...

else:
    def dump(obj, file, protocol=...):  # -> None:
        """Serialize obj as bytes streamed into file

        protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
        pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
        speed between processes running the same Python version.

        Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
        compatibility with older versions of Python.
        """
        ...
    def dumps(obj, protocol=...):  # -> bytes:
        """Serialize obj as a string of bytes allocated in memory

        protocol defaults to cloudpickle.DEFAULT_PROTOCOL which is an alias to
        pickle.HIGHEST_PROTOCOL. This setting favors maximum communication
        speed between processes running the same Python version.

        Set protocol=pickle.DEFAULT_PROTOCOL instead if you need to ensure
        compatibility with older versions of Python.
        """
        ...

class CloudPickler(Pickler):
    _dispatch_table = ...
    dispatch_table = ...
    def dump(self, obj): ...
    if pickle.HIGHEST_PROTOCOL >= 5:
        dispatch = ...
        def __init__(self, file, protocol=..., buffer_callback=...) -> None: ...
        def reducer_override(
            self, obj
        ):  # -> tuple[(origin: Unknown, args: Unknown) -> Unknown, tuple[Type[Literal] | None, Unknown] | tuple[Type[Final] | None, Unknown] | tuple[Type[ClassVar], Unknown] | tuple[Unknown, Unknown] | tuple[Type[Union], Unknown] | tuple[Type[Tuple[Unknown, ...]], Unknown] | tuple[(*args: Unknown, **kwargs: Unknown) -> Unknown, tuple[ellipsis | list[Unknown], Unknown]]] | tuple[Type[type], tuple[None]] | tuple[Type[type], tuple[ellipsis]] | tuple[Type[type], tuple[_NotImplementedType]] | tuple[(name: Unknown) -> (Type[type] | Any), tuple[Unknown]] | tuple[(bases: Unknown, name: Unknown, qualname: Unknown, members: Unknown, module: Unknown, class_tracker_id: Unknown, extra: Unknown) -> Unknown, tuple[Tuple[type, ...], str, str, dict[Unknown, Unknown], str, str | Unknown, None], tuple[dict[Unknown, Unknown], dict[Unknown, Unknown]], None, None, (obj: Unknown, state: Unknown) -> Unknown] | tuple[(type_constructor: Unknown, name: Unknown, bases: Unknown, type_kwargs: Unknown, class_tracker_id: Unknown, extra: Unknown) -> (Unknown | type), tuple[Any, Unknown, Any, dict[Unknown, Unknown], str | Unknown, None], tuple[dict[Unknown, Unknown], dict[Unknown, Unknown]], None, None, (obj: Unknown, state: Unknown) -> Unknown] | _NotImplementedType | tuple[Type[FunctionType], tuple[Unknown, Unknown, None, None, tuple[Any, ...] | None], tuple[Unknown, dict[str, Unknown]], None, None, (obj: Unknown, state: Unknown) -> None]:
            """Type-agnostic reducing callback for function and classes.

            For performance reasons, subclasses of the C _pickle.Pickler class
            cannot register custom reducers for functions and classes in the
            dispatch_table. Reducer for such types must instead implemented in
            the special reducer_override method.

            Note that method will be called for any object except a few
            builtin-types (int, lists, dicts etc.), which differs from reducers
            in the Pickler's dispatch_table, each of them being invoked for
            objects of a specific type only.

            This property comes in handy for classes: although most classes are
            instances of the ``type`` metaclass, some of them can be instances
            of other custom metaclasses (such as enum.EnumMeta for example). In
            particular, the metaclass will likely not be known in advance, and
            thus cannot be special-cased using an entry in the dispatch_table.
            reducer_override, among other things, allows us to register a
            reducer that will be called for any class, independently of its
            type.


            Notes:

            * reducer_override has the priority over dispatch_table-registered
            reducers.
            * reducer_override can be used to fix other limitations of
              cloudpickle for other types that suffered from type-specific
              reducers, such as Exceptions. See
              https://github.com/cloudpipe/cloudpickle/issues/248
            """
            ...
    else:
        dispatch = ...
        def __init__(self, file, protocol=...) -> None: ...
        def save_global(self, obj, name=..., pack=...):  # -> None:
            """
            Save a "global".

            The name of this method is somewhat misleading: all types get
            dispatched here.
            """
            ...
        def save_function(self, obj, name=...):  # -> None:
            """Registered with the dispatch to handle all function types.

            Determines what kind of function obj is (e.g. lambda, defined at
            interactive prompt, etc) and handles the pickling appropriately.
            """
            ...
        def save_pypy_builtin_func(self, obj):  # -> None:
            """Save pypy equivalent of builtin functions.
            PyPy does not have the concept of builtin-functions. Instead,
            builtin-functions are simple function instances, but with a
            builtin-code attribute.
            Most of the time, builtin functions should be pickled by attribute.
            But PyPy has flaky support for __qualname__, so some builtin
            functions such as float.__new__ will be classified as dynamic. For
            this reason only, we created this special routine. Because
            builtin-functions are not expected to have closure or globals,
            there is no additional hack (compared the one already implemented
            in pickle) to protect ourselves from reference cycles. A simple
            (reconstructor, newargs, obj.__dict__) tuple is save_reduced.  Note
            also that PyPy improved their support for __qualname__ in v3.6, so
            this routing should be removed when cloudpickle supports only PyPy
            3.6 and later.
            """
            ...
