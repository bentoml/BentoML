"""
Utilities for interpreting CSS from Stylers for formatting non-HTML outputs.
"""

class CSSWarning(UserWarning):
    """
    This CSS syntax cannot currently be parsed.
    """

    ...

class CSSResolver:
    """
    A callable for parsing and resolving CSS to atomic properties.
    """

    UNIT_RATIOS = ...
    FONT_SIZE_RATIOS = ...
    MARGIN_RATIOS = ...
    BORDER_WIDTH_RATIOS = ...
    SIDE_SHORTHANDS = ...
    SIDES = ...
    def __call__(
        self, declarations_str: str, inherited: dict[str, str] | None = ...
    ) -> dict[str, str]:
        """
        The given declarations to atomic properties.

        Parameters
        ----------
        declarations_str : str
            A list of CSS declarations
        inherited : dict, optional
            Atomic properties indicating the inherited style context in which
            declarations_str is to be resolved. ``inherited`` should already
            be resolved, i.e. valid output of this method.

        Returns
        -------
        dict
            Atomic CSS 2.2 properties.

        Examples
        --------
        >>> resolve = CSSResolver()
        >>> inherited = {'font-family': 'serif', 'font-weight': 'bold'}
        >>> out = resolve('''
        ...               border-color: BLUE RED;
        ...               font-size: 1em;
        ...               font-size: 2em;
        ...               font-weight: normal;
        ...               font-weight: inherit;
        ...               ''', inherited)
        >>> sorted(out.items())  # doctest: +NORMALIZE_WHITESPACE
        [('border-bottom-color', 'blue'),
         ('border-left-color', 'red'),
         ('border-right-color', 'red'),
         ('border-top-color', 'blue'),
         ('font-family', 'serif'),
         ('font-size', '24pt'),
         ('font-weight', 'bold')]
        """
        ...
    def size_to_pt(self, in_val, em_pt=..., conversions=...): ...
    def atomize(self, declarations): ...
    expand_border_color = ...
    expand_border_style = ...
    expand_border_width = ...
    expand_margin = ...
    expand_padding = ...
    def parse(self, declarations_str: str):  # -> Generator[tuple[str, str], None, None]:
        """
        Generates (prop, value) pairs from declarations.

        In a future version may generate parsed tokens from tinycss/tinycss2

        Parameters
        ----------
        declarations_str : str
        """
        ...
