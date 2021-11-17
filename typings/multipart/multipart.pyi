"""
This type stub file was generated by pyright.
"""

import typing as t

from .decoders import *
from .exceptions import *

_missing: object = ...
STATE_BEFORE_FIELD: int = ...
STATE_FIELD_NAME: int = ...
STATE_FIELD_DATA: int = ...
STATE_START: int = ...
STATE_START_BOUNDARY: int = ...
STATE_HEADER_FIELD_START: int = ...
STATE_HEADER_FIELD: int = ...
STATE_HEADER_VALUE_START: int = ...
STATE_HEADER_VALUE: int = ...
STATE_HEADER_VALUE_ALMOST_DONE: int = ...
STATE_HEADERS_ALMOST_DONE: int = ...
STATE_PART_DATA_START: int = ...
STATE_PART_DATA: int = ...
STATE_PART_DATA_END: int = ...
STATE_END: int = ...
STATES: t.List[str] = ...
FLAG_PART_BOUNDARY: int = ...
FLAG_LAST_BOUNDARY: int = ...
CR: int = ...
LF: int = ...
COLON: int = ...
SPACE: int = ...
HYPHEN: int = ...
AMPERSAND: int = ...
SEMICOLON: int = ...
LOWER_A: int = ...
LOWER_Z: int = ...
NULL: int = ...

lower_char: t.Callable[[int], int] = ...
ord_char: t.Callable[..., t.Any] = ...
join_bytes: t.Callable[[bytes], t.ByteString] = ...

SPECIAL_CHARS: t.ByteString = ...
QUOTED_STR: t.ByteString = ...
VALUE_STR: t.ByteString = ...
OPTION_RE_STR: t.ByteString = ...
OPTION_RE: t.ByteString = ...
QUOTE: t.ByteString = ...

def parse_options_header(
    value,
) -> t.Tuple[t.ByteString, t.Dict[t.ByteString, t.ByteString]] | tuple[
    bytes | Unknown, dict[Unknown, Unknown]
]:
    """
    Parses a Content-Type header into a value in the following format:
        (content_type, {parameters})
    """
    ...

class Field:
    """A Field object represents a (parsed) form field.  It represents a single
    field with a corresponding name and value.

    The name that a :class:`Field` will be instantiated with is the same name
    that would be found in the following HTML::

        <input name="name_goes_here" type="text"/>

    This class defines two methods, :meth:`on_data` and :meth:`on_end`, that
    will be called when data is written to the Field, and when the Field is
    finalized, respectively.

    :param name: the name of the form field
    """

    def __init__(self, name) -> None: ...
    @classmethod
    def from_value(klass, name, value):  # -> Self@Field:
        """Create an instance of a :class:`Field`, and set the corresponding
        value - either None or an actual value.  This method will also
        finalize the Field itself.

        :param name: the name of the form field
        :param value: the value of the form field - either a bytestring or
                      None
        """
        ...
    def write(self, data):  # -> int:
        """Write some data into the form field.

        :param data: a bytestring
        """
        ...
    def on_data(self, data):  # -> int:
        """This method is a callback that will be called whenever data is
        written to the Field.

        :param data: a bytestring
        """
        ...
    def on_end(self):  # -> None:
        """This method is called whenever the Field is finalized."""
        ...
    def finalize(self):  # -> None:
        """Finalize the form field."""
        ...
    def close(self):  # -> None:
        """Close the Field object.  This will free any underlying cache."""
        ...
    def set_none(self):  # -> None:
        """Some fields in a querystring can possibly have a value of None - for
        example, the string "foo&bar=&baz=asdf" will have a field with the
        name "foo" and value None, one with name "bar" and value "", and one
        with name "baz" and value "asdf".  Since the write() interface doesn't
        support writing None, this function will set the field value to None.
        """
        ...
    @property
    def field_name(self):  # -> Unknown:
        """This property returns the name of the field."""
        ...
    @property
    def value(self):  # -> bytes | object | None:
        """This property returns the value of the form field."""
        ...
    def __eq__(self, other) -> bool: ...
    def __repr__(self): ...

class File:
    """This class represents an uploaded file.  It handles writing file data to
    either an in-memory file or a temporary file on-disk, if the optional
    threshold is passed.

    There are some options that can be passed to the File to change behavior
    of the class.  Valid options are as follows:

    .. list-table::
       :widths: 15 5 5 30
       :header-rows: 1

       * - Name
         - Type
         - Default
         - Description
       * - UPLOAD_DIR
         - `str`
         - None
         - The directory to store uploaded files in.  If this is None, a
           temporary file will be created in the system's standard location.
       * - UPLOAD_DELETE_TMP
         - `bool`
         - True
         - Delete automatically created TMP file
       * - UPLOAD_KEEP_FILENAME
         - `bool`
         - False
         - Whether or not to keep the filename of the uploaded file.  If True,
           then the filename will be converted to a safe representation (e.g.
           by removing any invalid path segments), and then saved with the
           same name).  Otherwise, a temporary name will be used.
       * - UPLOAD_KEEP_EXTENSIONS
         - `bool`
         - False
         - Whether or not to keep the uploaded file's extension.  If False, the
           file will be saved with the default temporary extension (usually
           ".tmp").  Otherwise, the file's extension will be maintained.  Note
           that this will properly combine with the UPLOAD_KEEP_FILENAME
           setting.
       * - MAX_MEMORY_FILE_SIZE
         - `int`
         - 1 MiB
         - The maximum number of bytes of a File to keep in memory.  By
           default, the contents of a File are kept into memory until a certain
           limit is reached, after which the contents of the File are written
           to a temporary file.  This behavior can be disabled by setting this
           value to an appropriately large value (or, for example, infinity,
           such as `float('inf')`.

    :param file_name: The name of the file that this :class:`File` represents

    :param field_name: The field name that uploaded this file.  Note that this
                       can be None, if, for example, the file was uploaded
                       with Content-Type application/octet-stream

    :param config: The configuration for this File.  See above for valid
                   configuration keys and their corresponding values.
    """

    def __init__(self, file_name, field_name=..., config=...) -> None: ...
    @property
    def field_name(self):  # -> Unknown:
        """The form field associated with this file.  May be None if there isn't
        one, for example when we have an application/octet-stream upload.
        """
        ...
    @property
    def file_name(self):  # -> Unknown:
        """The file name given in the upload request."""
        ...
    @property
    def actual_file_name(self):  # -> bytes | None:
        """The file name that this file is saved as.  Will be None if it's not
        currently saved on disk.
        """
        ...
    @property
    def file_object(self):  # -> BytesIO | BufferedRandom | _TemporaryFileWrapper[str]:
        """The file object that we're currently writing to.  Note that this
        will either be an instance of a :class:`io.BytesIO`, or a regular file
        object.
        """
        ...
    @property
    def size(self):  # -> int:
        """The total size of this file, counted as the number of bytes that
        currently have been written to the file.
        """
        ...
    @property
    def in_memory(self):  # -> bool:
        """A boolean representing whether or not this file object is currently
        stored in-memory or on-disk.
        """
        ...
    def flush_to_disk(self):  # -> None:
        """If the file is already on-disk, do nothing.  Otherwise, copy from
        the in-memory buffer to a disk file, and then reassign our internal
        file object to this new disk file.

        Note that if you attempt to flush a file that is already on-disk, a
        warning will be logged to this module's logger.
        """
        ...
    def write(self, data):  # -> int:
        """Write some data to the File.

        :param data: a bytestring
        """
        ...
    def on_data(self, data):  # -> int:
        """This method is a callback that will be called whenever data is
        written to the File.

        :param data: a bytestring
        """
        ...
    def on_end(self):  # -> None:
        """This method is called whenever the Field is finalized."""
        ...
    def finalize(self):  # -> None:
        """Finalize the form file.  This will not close the underlying file,
        but simply signal that we are finished writing to the File.
        """
        ...
    def close(self):  # -> None:
        """Close the File object.  This will actually close the underlying
        file object (whether it's a :class:`io.BytesIO` or an actual file
        object).
        """
        ...
    def __repr__(self): ...

class BaseParser:
    """This class is the base class for all parsers.  It contains the logic for
    calling and adding callbacks.

    A callback can be one of two different forms.  "Notification callbacks" are
    callbacks that are called when something happens - for example, when a new
    part of a multipart message is encountered by the parser.  "Data callbacks"
    are called when we get some sort of data - for example, part of the body of
    a multipart chunk.  Notification callbacks are called with no parameters,
    whereas data callbacks are called with three, as follows::

        data_callback(data, start, end)

    The "data" parameter is a bytestring (i.e. "foo" on Python 2, or b"foo" on
    Python 3).  "start" and "end" are integer indexes into the "data" string
    that represent the data of interest.  Thus, in a data callback, the slice
    `data[start:end]` represents the data that the callback is "interested in".
    The callback is not passed a copy of the data, since copying severely hurts
    performance.
    """

    def __init__(self) -> None: ...
    def callback(self, name, data=..., start=..., end=...):  # -> None:
        """This function calls a provided callback with some data.  If the
        callback is not set, will do nothing.

        :param name: The name of the callback to call (as a string).

        :param data: Data to pass to the callback.  If None, then it is
                     assumed that the callback is a notification callback,
                     and no parameters are given.

        :param end: An integer that is passed to the data callback.

        :param start: An integer that is passed to the data callback.
        """
        ...
    def set_callback(self, name, new_func):  # -> None:
        """Update the function for a callback.  Removes from the callbacks dict
        if new_func is None.

        :param name: The name of the callback to call (as a string).

        :param new_func: The new function for the callback.  If None, then the
                         callback will be removed (with no error if it does not
                         exist).
        """
        ...
    def close(self): ...
    def finalize(self): ...
    def __repr__(self): ...

class OctetStreamParser(BaseParser):
    """This parser parses an octet-stream request body and calls callbacks when
    incoming data is received.  Callbacks are as follows:

    .. list-table::
       :widths: 15 10 30
       :header-rows: 1

       * - Callback Name
         - Parameters
         - Description
       * - on_start
         - None
         - Called when the first data is parsed.
       * - on_data
         - data, start, end
         - Called for each data chunk that is parsed.
       * - on_end
         - None
         - Called when the parser is finished parsing all data.

    :param callbacks: A dictionary of callbacks.  See the documentation for
                      :class:`BaseParser`.

    :param max_size: The maximum size of body to parse.  Defaults to infinity -
                     i.e. unbounded.
    """

    def __init__(self, callbacks=..., max_size=...) -> None: ...
    def write(self, data):  # -> int:
        """Write some data to the parser, which will perform size verification,
        and then pass the data to the underlying callback.

        :param data: a bytestring
        """
        ...
    def finalize(self):  # -> None:
        """Finalize this parser, which signals to that we are finished parsing,
        and sends the on_end callback.
        """
        ...
    def __repr__(self): ...

class QuerystringParser(BaseParser):
    """This is a streaming querystring parser.  It will consume data, and call
    the callbacks given when it has data.

    .. list-table::
       :widths: 15 10 30
       :header-rows: 1

       * - Callback Name
         - Parameters
         - Description
       * - on_field_start
         - None
         - Called when a new field is encountered.
       * - on_field_name
         - data, start, end
         - Called when a portion of a field's name is encountered.
       * - on_field_data
         - data, start, end
         - Called when a portion of a field's data is encountered.
       * - on_field_end
         - None
         - Called when the end of a field is encountered.
       * - on_end
         - None
         - Called when the parser is finished parsing all data.

    :param callbacks: A dictionary of callbacks.  See the documentation for
                      :class:`BaseParser`.

    :param strict_parsing: Whether or not to parse the body strictly.  Defaults
                           to False.  If this is set to True, then the behavior
                           of the parser changes as the following: if a field
                           has a value with an equal sign (e.g. "foo=bar", or
                           "foo="), it is always included.  If a field has no
                           equals sign (e.g. "...&name&..."), it will be
                           treated as an error if 'strict_parsing' is True,
                           otherwise included.  If an error is encountered,
                           then a
                           :class:`multipart.exceptions.QuerystringParseError`
                           will be raised.

    :param max_size: The maximum size of body to parse.  Defaults to infinity -
                     i.e. unbounded.
    """

    def __init__(self, callbacks=..., strict_parsing=..., max_size=...) -> None: ...
    def write(self, data):  # -> int:
        """Write some data to the parser, which will perform size verification,
        parse into either a field name or value, and then pass the
        corresponding data to the underlying callback.  If an error is
        encountered while parsing, a QuerystringParseError will be raised.  The
        "offset" attribute of the raised exception will be set to the offset in
        the input data chunk (NOT the overall stream) that caused the error.

        :param data: a bytestring
        """
        ...
    def finalize(self):  # -> None:
        """Finalize this parser, which signals to that we are finished parsing,
        if we're still in the middle of a field, an on_field_end callback, and
        then the on_end callback.
        """
        ...
    def __repr__(self): ...

class MultipartParser(BaseParser):
    """This class is a streaming multipart/form-data parser.

    .. list-table::
       :widths: 15 10 30
       :header-rows: 1

       * - Callback Name
         - Parameters
         - Description
       * - on_part_begin
         - None
         - Called when a new part of the multipart message is encountered.
       * - on_part_data
         - data, start, end
         - Called when a portion of a part's data is encountered.
       * - on_part_end
         - None
         - Called when the end of a part is reached.
       * - on_header_begin
         - None
         - Called when we've found a new header in a part of a multipart
           message
       * - on_header_field
         - data, start, end
         - Called each time an additional portion of a header is read (i.e. the
           part of the header that is before the colon; the "Foo" in
           "Foo: Bar").
       * - on_header_value
         - data, start, end
         - Called when we get data for a header.
       * - on_header_end
         - None
         - Called when the current header is finished - i.e. we've reached the
           newline at the end of the header.
       * - on_headers_finished
         - None
         - Called when all headers are finished, and before the part data
           starts.
       * - on_end
         - None
         - Called when the parser is finished parsing all data.


    :param boundary: The multipart boundary.  This is required, and must match
                     what is given in the HTTP request - usually in the
                     Content-Type header.

    :param callbacks: A dictionary of callbacks.  See the documentation for
                      :class:`BaseParser`.

    :param max_size: The maximum size of body to parse.  Defaults to infinity -
                     i.e. unbounded.
    """

    def __init__(self, boundary, callbacks=..., max_size=...) -> None: ...
    def write(self, data):  # -> int:
        """Write some data to the parser, which will perform size verification,
        and then parse the data into the appropriate location (e.g. header,
        data, etc.), and pass this on to the underlying callback.  If an error
        is encountered, a MultipartParseError will be raised.  The "offset"
        attribute on the raised exception will be set to the offset of the byte
        in the input chunk that caused the error.

        :param data: a bytestring
        """
        ...
    def finalize(self):  # -> None:
        """Finalize this parser, which signals to that we are finished parsing.

        Note: It does not currently, but in the future, it will verify that we
        are in the final state of the parser (i.e. the end of the multipart
        message is well-formed), and, if not, throw an error.
        """
        ...
    def __repr__(self): ...

class FormParser:
    """This class is the all-in-one form parser.  Given all the information
    necessary to parse a form, it will instantiate the correct parser, create
    the proper :class:`Field` and :class:`File` classes to store the data that
    is parsed, and call the two given callbacks with each field and file as
    they become available.

    :param content_type: The Content-Type of the incoming request.  This is
                         used to select the appropriate parser.

    :param on_field: The callback to call when a field has been parsed and is
                     ready for usage.  See above for parameters.

    :param on_file: The callback to call when a file has been parsed and is
                    ready for usage.  See above for parameters.

    :param on_end: An optional callback to call when all fields and files in a
                   request has been parsed.  Can be None.

    :param boundary: If the request is a multipart/form-data request, this
                     should be the boundary of the request, as given in the
                     Content-Type header, as a bytestring.

    :param file_name: If the request is of type application/octet-stream, then
                      the body of the request will not contain any information
                      about the uploaded file.  In such cases, you can provide
                      the file name of the uploaded file manually.

    :param FileClass: The class to use for uploaded files.  Defaults to
                      :class:`File`, but you can provide your own class if you
                      wish to customize behaviour.  The class will be
                      instantiated as FileClass(file_name, field_name), and it
                      must provide the folllowing functions::
                          file_instance.write(data)
                          file_instance.finalize()
                          file_instance.close()

    :param FieldClass: The class to use for uploaded fields.  Defaults to
                       :class:`Field`, but you can provide your own class if
                       you wish to customize behaviour.  The class will be
                       instantiated as FieldClass(field_name), and it must
                       provide the folllowing functions::
                           field_instance.write(data)
                           field_instance.finalize()
                           field_instance.close()

    :param config: Configuration to use for this FormParser.  The default
                   values are taken from the DEFAULT_CONFIG value, and then
                   any keys present in this dictionary will overwrite the
                   default values.

    """

    DEFAULT_CONFIG = ...
    def __init__(
        self,
        content_type,
        on_field,
        on_file,
        on_end=...,
        boundary=...,
        file_name=...,
        FileClass=...,
        FieldClass=...,
        config=...,
    ) -> None: ...
    def write(self, data):  # -> int:
        """Write some data.  The parser will forward this to the appropriate
        underlying parser.

        :param data: a bytestring
        """
        ...
    def finalize(self):  # -> None:
        """Finalize the parser."""
        ...
    def close(self):  # -> None:
        """Close the parser."""
        ...
    def __repr__(self): ...

def create_form_parser(
    headers, on_field, on_file, trust_x_headers=..., config=...
):  # -> FormParser:
    """This function is a helper function to aid in creating a FormParser
    instances.  Given a dictionary-like headers object, it will determine
    the correct information needed, instantiate a FormParser with the
    appropriate values and given callbacks, and then return the corresponding
    parser.

    :param headers: A dictionary-like object of HTTP headers.  The only
                    required header is Content-Type.

    :param on_field: Callback to call with each parsed field.

    :param on_file: Callback to call with each parsed file.

    :param trust_x_headers: Whether or not to trust information received from
                            certain X-Headers - for example, the file name from
                            X-File-Name.

    :param config: Configuration variables to pass to the FormParser.
    """
    ...

def parse_form(
    headers, input_stream, on_field, on_file, chunk_size=..., **kwargs
):  # -> None:
    """This function is useful if you just want to parse a request body,
    without too much work.  Pass it a dictionary-like object of the request's
    headers, and a file-like object for the input stream, along with two
    callbacks that will get called whenever a field or file is parsed.

    :param headers: A dictionary-like object of HTTP headers.  The only
                    required header is Content-Type.

    :param input_stream: A file-like object that represents the request body.
                         The read() method must return bytestrings.

    :param on_field: Callback to call with each parsed field.

    :param on_file: Callback to call with each parsed file.

    :param chunk_size: The maximum size to read from the input stream and write
                       to the parser at one time.  Defaults to 1 MiB.
    """
    ...
