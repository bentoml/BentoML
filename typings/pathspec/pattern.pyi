

"""
This module provides the base definition for patterns.
"""

class Pattern:
    """
    The :class:`Pattern` class is the abstract definition of a pattern.
    """

    __slots__ = ...
    def __init__(self, include: Optional[bool]) -> None:
        """
        Initializes the :class:`Pattern` instance.

        *include* (:class:`bool` or :data:`None`) is whether the matched
        files should be included (:data:`True`), excluded (:data:`False`),
        or is a null-operation (:data:`None`).
        """
        ...
    def match(self, files: Iterable[Text]) -> Iterator[Text]:
        """
        Matches this pattern against the specified files.

        *files* (:class:`~collections.abc.Iterable` of :class:`str`) contains
        each file relative to the root directory (e.g., ``"relative/path/to/file"``).

        Returns an :class:`~collections.abc.Iterable` yielding each matched
        file path (:class:`str`).
        """
        ...

class RegexPattern(Pattern):
    """
    The :class:`RegexPattern` class is an implementation of a pattern
    using regular expressions.
    """

    __slots__ = ...
    def __init__(
        self, pattern: Union[AnyStr, RegexHint], include: Optional[bool] = ...
    ) -> None:
        """
        Initializes the :class:`RegexPattern` instance.

        *pattern* (:class:`unicode`, :class:`bytes`, :class:`re.RegexObject`,
        or :data:`None`) is the pattern to compile into a regular
        expression.

        *include* (:class:`bool` or :data:`None`) must be :data:`None`
        unless *pattern* is a precompiled regular expression (:class:`re.RegexObject`)
        in which case it is whether matched files should be included
        (:data:`True`), excluded (:data:`False`), or is a null operation
        (:data:`None`).

                .. NOTE:: Subclasses do not need to support the *include*
                   parameter.
        """
        ...
    def __eq__(self, other: RegexPattern) -> bool:
        """
        Tests the equality of this regex pattern with *other* (:class:`RegexPattern`)
        by comparing their :attr:`~Pattern.include` and :attr:`~RegexPattern.regex`
        attributes.
        """
        ...
    def match(self, files: Iterable[Text]) -> Iterable[Text]:
        """
        Matches this pattern against the specified files.

        *files* (:class:`~collections.abc.Iterable` of :class:`str`)
        contains each file relative to the root directory (e.g., "relative/path/to/file").

        Returns an :class:`~collections.abc.Iterable` yielding each matched
        file path (:class:`str`).
        """
        ...
    @classmethod
    def pattern_to_regex(cls, pattern: Text) -> Tuple[Text, bool]:
        """
        Convert the pattern into an uncompiled regular expression.

        *pattern* (:class:`str`) is the pattern to convert into a regular
        expression.

        Returns the uncompiled regular expression (:class:`str` or :data:`None`),
        and whether matched files should be included (:data:`True`),
        excluded (:data:`False`), or is a null-operation (:data:`None`).

                .. NOTE:: The default implementation simply returns *pattern* and
                   :data:`True`.
        """
        ...
