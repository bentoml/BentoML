

from ..pattern import RegexPattern

"""
This module implements Git's wildmatch pattern matching which itself is
derived from Rsync's wildmatch. Git uses wildmatch for its ".gitignore"
files.
"""
_BYTES_ENCODING = ...

class GitWildMatchPatternError(ValueError):
    """
    The :class:`GitWildMatchPatternError` indicates an invalid git wild match
    pattern.
    """

    ...

class GitWildMatchPattern(RegexPattern):
    """
    The :class:`GitWildMatchPattern` class represents a compiled Git
    wildmatch pattern.
    """

    __slots__ = ...
    @classmethod
    def pattern_to_regex(
        cls, pattern: AnyStr
    ) -> Tuple[Optional[AnyStr], Optional[bool]]:
        """
        Convert the pattern into a regular expression.

        *pattern* (:class:`unicode` or :class:`bytes`) is the pattern to
        convert into a regular expression.

        Returns the uncompiled regular expression (:class:`unicode`, :class:`bytes`,
        or :data:`None`), and whether matched files should be included
        (:data:`True`), excluded (:data:`False`), or if it is a
        null-operation (:data:`None`).
        """
        ...
    @staticmethod
    def escape(s: AnyStr) -> AnyStr:
        """
        Escape special characters in the given string.

        *s* (:class:`unicode` or :class:`bytes`) a filename or a string
        that you want to escape, usually before adding it to a `.gitignore`

        Returns the escaped string (:class:`unicode` or :class:`bytes`)
        """
        ...

class GitIgnorePattern(GitWildMatchPattern):
    """
    The :class:`GitIgnorePattern` class is deprecated by :class:`GitWildMatchPattern`.
    This class only exists to maintain compatibility with v0.4.
    """

    def __init__(self, *args, **kw) -> None:
        """
        Warn about deprecation.
        """
        ...
    @classmethod
    def pattern_to_regex(cls, *args, **kw):  # -> Tuple[Unknown | None, bool | None]:
        """
        Warn about deprecation.
        """
        ...
