class Flags:
    """
    Flags that apply to pandas objects.

    .. versionadded:: 1.2.0

    Parameters
    ----------
    obj : Series or DataFrame
        The object these flags are associated with.
    allows_duplicate_labels : bool, default True
        Whether to allow duplicate labels in this object. By default,
        duplicate labels are permitted. Setting this to ``False`` will
        cause an :class:`errors.DuplicateLabelError` to be raised when
        `index` (or columns for DataFrame) is not unique, or any
        subsequent operation on introduces duplicates.
        See :ref:`duplicates.disallow` for more.

        .. warning::

           This is an experimental feature. Currently, many methods fail to
           propagate the ``allows_duplicate_labels`` value. In future versions
           it is expected that every method taking or returning one or more
           DataFrame or Series objects will propagate ``allows_duplicate_labels``.

    Notes
    -----
    Attributes can be set in two ways

    >>> df = pd.DataFrame()
    >>> df.flags
    <Flags(allows_duplicate_labels=True)>
    >>> df.flags.allows_duplicate_labels = False
    >>> df.flags
    <Flags(allows_duplicate_labels=False)>

    >>> df.flags['allows_duplicate_labels'] = True
    >>> df.flags
    <Flags(allows_duplicate_labels=True)>
    """

    _keys = ...
    def __init__(self, obj, *, allows_duplicate_labels) -> None: ...
    @property
    def allows_duplicate_labels(self) -> bool:
        """
        Whether this object allows duplicate labels.

        Setting ``allows_duplicate_labels=False`` ensures that the
        index (and columns of a DataFrame) are unique. Most methods
        that accept and return a Series or DataFrame will propagate
        the value of ``allows_duplicate_labels``.

        See :ref:`duplicates` for more.

        See Also
        --------
        DataFrame.attrs : Set global metadata on this object.
        DataFrame.set_flags : Set global flags on this object.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2]}, index=['a', 'a'])
        >>> df.allows_duplicate_labels
        True
        >>> df.allows_duplicate_labels = False
        Traceback (most recent call last):
            ...
        pandas.errors.DuplicateLabelError: Index has duplicates.
              positions
        label
        a        [0, 1]
        """
        ...
    @allows_duplicate_labels.setter
    def allows_duplicate_labels(self, value: bool): ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value): ...
    def __repr__(self): ...
    def __eq__(self, other) -> bool: ...
