import os
import typing as t


def resolve_user_filepath(filepath: str, ctx: t.Optional[str]) -> str:
    """Resolve the abspath of a filepath provided by user, which may contain "~" or may
    be a relative path base on ctx dir.
    """
    # Return if filepath exist after expanduser
    _path = os.path.expanduser(filepath)
    if os.path.exists(_path):
        return os.path.realpath(_path)

    # Try finding file in ctx if provided
    if ctx:
        _path = os.path.expanduser(os.path.join(ctx, filepath))
        if os.path.exists(_path):
            return os.path.realpath(_path)

    raise FileNotFoundError(f"file {filepath} not found")
