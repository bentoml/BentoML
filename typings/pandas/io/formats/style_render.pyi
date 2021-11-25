from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pandas import DataFrame, Index
from pandas._typing import FrameOrSeriesUnion, TypedDict

jinja2 = ...
BaseFormatter = Union[str, Callable]
ExtFormatter = Union[BaseFormatter, Dict[Any, Optional[BaseFormatter]]]
CSSPair = Tuple[str, Union[str, int, float]]
CSSList = List[CSSPair]
CSSProperties = Union[str, CSSList]

class CSSDict(TypedDict):
    selector: str
    props: CSSProperties
    ...

CSSStyles = List[CSSDict]
Subset = Union[slice, Sequence, Index]

class StylerRenderer:
    """
    Base class to process rendering a Styler with a specified jinja2 template.
    """

    loader = ...
    env = ...
    template_html = ...
    template_html_table = ...
    template_html_style = ...
    template_latex = ...
    def __init__(
        self,
        data: FrameOrSeriesUnion,
        uuid: str | None = ...,
        uuid_len: int = ...,
        table_styles: CSSStyles | None = ...,
        table_attributes: str | None = ...,
        caption: str | tuple | None = ...,
        cell_ids: bool = ...,
    ) -> None: ...
    def format(
        self,
        formatter: ExtFormatter | None = ...,
        subset: Subset | None = ...,
        na_rep: str | None = ...,
        precision: int | None = ...,
        decimal: str = ...,
        thousands: str | None = ...,
        escape: str | None = ...,
    ) -> StylerRenderer:
        r"""
        Format the text display value of cells.

        Parameters
        ----------
        formatter : str, callable, dict or None
            Object to define how values are displayed. See notes.
        subset : label, array-like, IndexSlice, optional
            A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
            or single key, to `DataFrame.loc[:, <subset>]` where the columns are
            prioritised, to limit ``data`` to *before* applying the function.
        na_rep : str, optional
            Representation for missing values.
            If ``na_rep`` is None, no special formatting is applied.

            .. versionadded:: 1.0.0

        precision : int, optional
            Floating point precision to use for display purposes, if not determined by
            the specified ``formatter``.

            .. versionadded:: 1.3.0

        decimal : str, default "."
            Character used as decimal separator for floats, complex and integers

            .. versionadded:: 1.3.0

        thousands : str, optional, default None
            Character used as thousands separator for floats, complex and integers

            .. versionadded:: 1.3.0

        escape : str, optional
            Use 'html' to replace the characters ``&``, ``<``, ``>``, ``'``, and ``"``
            in cell display string with HTML-safe sequences.
            Use 'latex' to replace the characters ``&``, ``%``, ``$``, ``#``, ``_``,
            ``{``, ``}``, ``~``, ``^``, and ``\`` in the cell display string with
            LaTeX-safe sequences.
            Escaping is done before ``formatter``.

            .. versionadded:: 1.3.0

        Returns
        -------
        self : Styler

        Notes
        -----
        This method assigns a formatting function, ``formatter``, to each cell in the
        DataFrame. If ``formatter`` is ``None``, then the default formatter is used.
        If a callable then that function should take a data value as input and return
        a displayable representation, such as a string. If ``formatter`` is
        given as a string this is assumed to be a valid Python format specification
        and is wrapped to a callable as ``string.format(x)``. If a ``dict`` is given,
        keys should correspond to column names, and values should be string or
        callable, as above.

        The default formatter currently expresses floats and complex numbers with the
        pandas display precision unless using the ``precision`` argument here. The
        default formatter does not adjust the representation of missing values unless
        the ``na_rep`` argument is used.

        The ``subset`` argument defines which region to apply the formatting function
        to. If the ``formatter`` argument is given in dict form but does not include
        all columns within the subset then these columns will have the default formatter
        applied. Any columns in the formatter dict excluded from the subset will
        be ignored.

        When using a ``formatter`` string the dtypes must be compatible, otherwise a
        `ValueError` will be raised.

        Examples
        --------
        Using ``na_rep`` and ``precision`` with the default ``formatter``

        >>> df = pd.DataFrame([[np.nan, 1.0, 'A'], [2.0, np.nan, 3.0]])
        >>> df.style.format(na_rep='MISS', precision=3)
                0       1       2
        0    MISS   1.000       A
        1   2.000    MISS   3.000

        Using a ``formatter`` specification on consistent column dtypes

        >>> df.style.format('{:.2f}', na_rep='MISS', subset=[0,1])
                0      1          2
        0    MISS   1.00          A
        1    2.00   MISS   3.000000

        Using the default ``formatter`` for unspecified columns

        >>> df.style.format({0: '{:.2f}', 1: '£ {:.1f}'}, na_rep='MISS', precision=1)
                 0      1     2
        0    MISS   £ 1.0     A
        1    2.00    MISS   3.0

        Multiple ``na_rep`` or ``precision`` specifications under the default
        ``formatter``.

        >>> df.style.format(na_rep='MISS', precision=1, subset=[0])
        ...     .format(na_rep='PASS', precision=2, subset=[1, 2])
                0      1      2
        0    MISS   1.00      A
        1     2.0   PASS   3.00

        Using a callable ``formatter`` function.

        >>> func = lambda s: 'STRING' if isinstance(s, str) else 'FLOAT'
        >>> df.style.format({0: '{:.1f}', 2: func}, precision=4, na_rep='MISS')
                0        1        2
        0    MISS   1.0000   STRING
        1     2.0     MISS    FLOAT

        Using a ``formatter`` with HTML ``escape`` and ``na_rep``.

        >>> df = pd.DataFrame([['<div></div>', '"A&B"', None]])
        >>> s = df.style.format(
        ...     '<a href="a.com/{0}">{0}</a>', escape="html", na_rep="NA"
        ...     )
        >>> s.render()
        ...
        <td .. ><a href="a.com/&lt;div&gt;&lt;/div&gt;">&lt;div&gt;&lt;/div&gt;</a></td>
        <td .. ><a href="a.com/&#34;A&amp;B&#34;">&#34;A&amp;B&#34;</a></td>
        <td .. >NA</td>
        ...

        Using a ``formatter`` with LaTeX ``escape``.

        >>> df = pd.DataFrame([["123"], ["~ ^"], ["$%#"]])
        >>> s = df.style.format("\\textbf{{{}}}", escape="latex").to_latex()
        \begin{tabular}{ll}
        {} & {0} \\
        0 & \textbf{123} \\
        1 & \textbf{\textasciitilde \space \textasciicircum } \\
        2 & \textbf{\$\%\#} \\
        \end{tabular}
        """
        ...

def non_reducing_slice(slice_: Subset):  # -> tuple[Unknown | list[Unknown], ...]:
    """
    Ensure that a slice doesn't reduce to a Series or Scalar.

    Any user-passed `subset` should have this called on it
    to make sure we're always working with DataFrames.
    """
    ...

def maybe_convert_css_to_tuples(style: CSSProperties) -> CSSList:
    """
    Convert css-string to sequence of tuples format if needed.
    'color:red; border:1px solid black;' -> [('color', 'red'),
                                             ('border','1px solid red')]
    """
    ...

class Tooltips:
    """
    An extension to ``Styler`` that allows for and manipulates tooltips on hover
    of ``<td>`` cells in the HTML result.

    Parameters
    ----------
    css_name: str, default "pd-t"
        Name of the CSS class that controls visualisation of tooltips.
    css_props: list-like, default; see Notes
        List of (attr, value) tuples defining properties of the CSS class.
    tooltips: DataFrame, default empty
        DataFrame of strings aligned with underlying Styler data for tooltip
        display.

    Notes
    -----
    The default properties for the tooltip CSS class are:

        - visibility: hidden
        - position: absolute
        - z-index: 1
        - background-color: black
        - color: white
        - transform: translate(-20px, -20px)

    Hidden visibility is a key prerequisite to the hover functionality, and should
    always be included in any manual properties specification.
    """

    def __init__(
        self,
        css_props: CSSProperties = ...,
        css_name: str = ...,
        tooltips: DataFrame = ...,
    ) -> None: ...
