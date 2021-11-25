

class FormParserError(ValueError):
    """Base error class for our form parser."""

    ...

class ParseError(FormParserError):
    """This exception (or a subclass) is raised when there is an error while
    parsing something.
    """

    offset: int = ...

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

class FileError(FormParserError, OSError):
    """Exception class for problems with the File class."""

    ...

Base64Error: TypeError = ...
