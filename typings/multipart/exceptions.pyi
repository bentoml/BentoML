"""
This type stub file was generated by pyright.
"""

from six import PY3

class FormParserError(ValueError):
    """Base error class for our form parser."""

    ...

class ParseError(FormParserError):
    """This exception (or a subclass) is raised when there is an error while
    parsing something.
    """

    offset = ...

class MultipartParseError(ParseError):
    """This is a specific error that is raised when the MultipartParser detects
    an error while parsing.
    """

    ...

class QuerystringParseError(ParseError):
    """This is a specific error that is raised when the QuerystringParser
    detects an error while parsing.
    """

    ...

class DecodeError(ParseError):
    """This exception is raised when there is a decoding error - for example
    with the Base64Decoder or QuotedPrintableDecoder.
    """

    ...

if IOError is not OSError:
    class FileError(FormParserError, IOError, OSError):
        """Exception class for problems with the File class."""

        ...

else:
    class FileError(FormParserError, OSError):
        """Exception class for problems with the File class."""

        ...

if PY3:
    Base64Error = ...
else:
    Base64Error = ...
