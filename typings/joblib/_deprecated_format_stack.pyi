"""
Represent an exception with a lot of information.

Provides 2 useful functions:

format_exc: format an exception into a complete traceback, with full
            debugging instruction.

format_outer_frames: format the current position in the stack call.

Adapted from IPython's VerboseTB.
"""
INDENT = ...

def safe_repr(value):  # -> str | None:
    """Hopefully pretty robust repr equivalent."""
    ...

def eq_repr(value, repr=...): ...
def uniq_stable(elems):  # -> list[Unknown]:
    """uniq_stable(elems) -> list

    Return from an iterable, a list of all the unique elements in the input,
    but maintaining the order in which they first appear.

    A naive solution to this problem which just makes a dictionary with the
    elements as keys fails to respect the stability condition, since
    dictionaries are unsorted by nature.

    Note: All elements in the input must be hashable.
    """
    ...

def fix_frame_records_filenames(records):  # -> list[Unknown]:
    """Try to fix the filenames in each record from inspect.getinnerframes().

    Particularly, modules loaded from within zip files have useless filenames
    attached to their code object, and inspect.getinnerframes() just uses it.
    """
    ...

def format_records(records): ...
def format_exc(etype, evalue, etb, context=..., tb_offset=...):  # -> str:
    """Return a nice text document describing the traceback.

    Parameters
    -----------
    etype, evalue, etb: as returned by sys.exc_info
    context: number of lines of the source file to plot
    tb_offset: the number of stack frame not to use (0 = use all)

    """
    ...

def format_outer_frames(
    context=..., stack_start=..., stack_end=..., ignore_ipython=...
): ...
