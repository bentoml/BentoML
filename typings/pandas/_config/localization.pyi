from contextlib import contextmanager

"""
Helpers for configuring locale settings.

Name `localization` is chosen to avoid overlap with builtin `locale` module.
"""

@contextmanager
def set_locale(new_locale, lc_var: int = ...):  # -> Generator[str | Unknown, None, None]:
    """
    Context manager for temporarily setting a locale.

    Parameters
    ----------
    new_locale : str or tuple
        A string of the form <language_country>.<encoding>. For example to set
        the current locale to US English with a UTF8 encoding, you would pass
        "en_US.UTF-8".
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Notes
    -----
    This is useful when you want to run a particular block of code under a
    particular locale, without globally setting the locale. This probably isn't
    thread-safe.
    """
    ...

def can_set_locale(lc: str, lc_var: int = ...) -> bool:
    """
    Check to see if we can set a locale, and subsequently get the locale,
    without raising an Exception.

    Parameters
    ----------
    lc : str
        The locale to attempt to set.
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Returns
    -------
    bool
        Whether the passed locale can be set
    """
    ...

def get_locales(prefix=..., normalize=..., locale_getter=...):
    """
    Get all the locales that are available on the system.

    Parameters
    ----------
    prefix : str
        If not ``None`` then return only those locales with the prefix
        provided. For example to get all English language locales (those that
        start with ``"en"``), pass ``prefix="en"``.
    normalize : bool
        Call ``locale.normalize`` on the resulting list of available locales.
        If ``True``, only locales that can be set without throwing an
        ``Exception`` are returned.
    locale_getter : callable
        The function to use to retrieve the current locales. This should return
        a string with each locale separated by a newline character.

    Returns
    -------
    locales : list of strings
        A list of locale strings that can be set with ``locale.setlocale()``.
        For example::

            locale.setlocale(locale.LC_ALL, locale_string)

    On error will return None (no locale available, e.g. Windows)

    """
    ...
