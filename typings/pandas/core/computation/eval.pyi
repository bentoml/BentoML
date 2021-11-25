from pandas.core.computation.ops import BinOp

"""
Top level ``eval`` module.
"""

def eval(
    expr: str | BinOp,
    parser: str = ...,
    engine: str | None = ...,
    truediv=...,
    local_dict=...,
    global_dict=...,
    resolvers=...,
    level=...,
    target=...,
    inplace=...,
):  # -> object | None:
    """
    Evaluate a Python expression as a string using various backends.

    The following arithmetic operations are supported: ``+``, ``-``, ``*``,
    ``/``, ``**``, ``%``, ``//`` (python engine only) along with the following
    boolean operations: ``|`` (or), ``&`` (and), and ``~`` (not).
    Additionally, the ``'pandas'`` parser allows the use of :keyword:`and`,
    :keyword:`or`, and :keyword:`not` with the same semantics as the
    corresponding bitwise operators.  :class:`~pandas.Series` and
    :class:`~pandas.DataFrame` objects are supported and behave as they would
    with plain ol' Python evaluation.

    Parameters
    ----------
    expr : str
        The expression to evaluate. This string cannot contain any Python
        `statements
        <https://docs.python.org/3/reference/simple_stmts.html#simple-statements>`__,
        only Python `expressions
        <https://docs.python.org/3/reference/simple_stmts.html#expression-statements>`__.
    parser : {'pandas', 'python'}, default 'pandas'
        The parser to use to construct the syntax tree from the expression. The
        default of ``'pandas'`` parses code slightly different than standard
        Python. Alternatively, you can parse an expression using the
        ``'python'`` parser to retain strict Python semantics.  See the
        :ref:`enhancing performance <enhancingperf.eval>` documentation for
        more details.
    engine : {'python', 'numexpr'}, default 'numexpr'

        The engine used to evaluate the expression. Supported engines are

        - None         : tries to use ``numexpr``, falls back to ``python``
        - ``'numexpr'``: This default engine evaluates pandas objects using
                         numexpr for large speed ups in complex expressions
                         with large frames.
        - ``'python'``: Performs operations as if you had ``eval``'d in top
                        level python. This engine is generally not that useful.

        More backends may be available in the future.

    truediv : bool, optional
        Whether to use true division, like in Python >= 3.

        .. deprecated:: 1.0.0

    local_dict : dict or None, optional
        A dictionary of local variables, taken from locals() by default.
    global_dict : dict or None, optional
        A dictionary of global variables, taken from globals() by default.
    resolvers : list of dict-like or None, optional
        A list of objects implementing the ``__getitem__`` special method that
        you can use to inject an additional collection of namespaces to use for
        variable lookup. For example, this is used in the
        :meth:`~DataFrame.query` method to inject the
        ``DataFrame.index`` and ``DataFrame.columns``
        variables that refer to their respective :class:`~pandas.DataFrame`
        instance attributes.
    level : int, optional
        The number of prior stack frames to traverse and add to the current
        scope. Most users will **not** need to change this parameter.
    target : object, optional, default None
        This is the target object for assignment. It is used when there is
        variable assignment in the expression. If so, then `target` must
        support item assignment with string keys, and if a copy is being
        returned, it must also support `.copy()`.
    inplace : bool, default False
        If `target` is provided, and the expression mutates `target`, whether
        to modify `target` inplace. Otherwise, return a copy of `target` with
        the mutation.

    Returns
    -------
    ndarray, numeric scalar, DataFrame, Series, or None
        The completion value of evaluating the given code or None if ``inplace=True``.

    Raises
    ------
    ValueError
        There are many instances where such an error can be raised:

        - `target=None`, but the expression is multiline.
        - The expression is multiline, but not all them have item assignment.
          An example of such an arrangement is this:

          a = b + 1
          a + 2

          Here, there are expressions on different lines, making it multiline,
          but the last line has no variable assigned to the output of `a + 2`.
        - `inplace=True`, but the expression is missing item assignment.
        - Item assignment is provided, but the `target` does not support
          string item assignment.
        - Item assignment is provided and `inplace=False`, but the `target`
          does not support the `.copy()` method

    See Also
    --------
    DataFrame.query : Evaluates a boolean expression to query the columns
            of a frame.
    DataFrame.eval : Evaluate a string describing operations on
            DataFrame columns.

    Notes
    -----
    The ``dtype`` of any objects involved in an arithmetic ``%`` operation are
    recursively cast to ``float64``.

    See the :ref:`enhancing performance <enhancingperf.eval>` documentation for
    more details.

    Examples
    --------
    >>> df = pd.DataFrame({"animal": ["dog", "pig"], "age": [10, 20]})
    >>> df
      animal  age
    0    dog   10
    1    pig   20

    We can add a new column using ``pd.eval``:

    >>> pd.eval("double_age = df.age * 2", target=df)
      animal  age  double_age
    0    dog   10          20
    1    pig   20          40
    """
    ...
