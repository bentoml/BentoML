def ensure_decoded(s):  # -> str:
    """
    If we have bytes, decode them to unicode.
    """
    ...

def result_type_many(*arrays_and_dtypes):
    """
    Wrapper around numpy.result_type which overcomes the NPY_MAXARGS (32)
    argument limit.
    """
    ...
