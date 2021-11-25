""" io on the clipboard """

def read_clipboard(sep=..., **kwargs):
    r"""
    Read text from clipboard and pass to read_csv.

    Parameters
    ----------
    sep : str, default '\s+'
        A string or regex delimiter. The default of '\s+' denotes
        one or more whitespace characters.

    **kwargs
        See read_csv for the full argument list.

    Returns
    -------
    DataFrame
        A parsed DataFrame object.
    """
    ...

def to_clipboard(obj, excel=..., sep=..., **kwargs):
    """
    Attempt to write text representation of object to the system clipboard
    The clipboard can be then pasted into Excel for example.

    Parameters
    ----------
    obj : the object to write to the clipboard
    excel : bool, defaults to True
            if True, use the provided separator, writing in a csv
            format for allowing easy pasting into excel.
            if False, write a string representation of the object
            to the clipboard
    sep : optional, defaults to tab
    other keywords are passed to to_csv

    Notes
    -----
    Requirements for your platform
      - Linux: xclip, or xsel (with PyQt4 modules)
      - Windows:
      - OS X:
    """
    ...
