import contextlib

@contextlib.contextmanager
def rewrite_exception(old_name: str, new_name: str):  # -> Generator[None, None, None]:
    """
    Rewrite the message of an exception.
    """
    ...

def find_stack_level() -> int:
    """
    Find the first place in the stack that is not inside pandas
    (tests notwithstanding).
    """
    ...
