import io

from pandas._typing import Buffer, CompressionOptions, FilePathOrBuffer, StorageOptions
from pandas.core.frame import DataFrame
from pandas.core.shared_docs import _shared_docs
from pandas.util._decorators import doc

"""
:mod:`pandas.io.xml` is a module for reading XML.
"""

class _XMLFrameParser:
    """
    Internal subclass to parse XML into DataFrames.

    Parameters
    ----------
    path_or_buffer : a valid JSON str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid
        URL schemes include http, ftp, s3, and file.

    xpath : str or regex
        The XPath expression to parse required set of nodes for
        migration to `Data Frame`. `etree` supports limited XPath.

    namespacess : dict
        The namespaces defined in XML document (`xmlns:namespace='URI')
        as dicts with key being namespace and value the URI.

    elems_only : bool
        Parse only the child elements at the specified `xpath`.

    attrs_only : bool
        Parse only the attributes at the specified `xpath`.

    names : list
        Column names for Data Frame of parsed XML data.

    encoding : str
        Encoding of xml object or document.

    stylesheet : str or file-like
        URL, file, file-like object, or a raw string containing XSLT,
        `etree` does not support XSLT but retained for consistency.

    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, default 'infer'
        Compression type for on-the-fly decompression of on-disk data.
        If 'infer', then use extension for gzip, bz2, zip or xz.

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection,
        e.g. host, port, username, password, etc.,

    See also
    --------
    pandas.io.xml._EtreeFrameParser
    pandas.io.xml._LxmlFrameParser

    Notes
    -----
    To subclass this class effectively you must override the following methods:`
        * :func:`parse_data`
        * :func:`_parse_nodes`
        * :func:`_parse_doc`
        * :func:`_validate_names`
        * :func:`_validate_path`


    See each method's respective documentation for details on their
    functionality.
    """

    def __init__(
        self,
        path_or_buffer,
        xpath,
        namespaces,
        elems_only,
        attrs_only,
        names,
        encoding,
        stylesheet,
        compression,
        storage_options,
    ) -> None: ...
    def parse_data(self) -> list[dict[str, str | None]]:
        """
        Parse xml data.

        This method will call the other internal methods to
        validate xpath, names, parse and return specific nodes.
        """
        ...

class _EtreeFrameParser(_XMLFrameParser):
    """
    Internal class to parse XML into DataFrames with the Python
    standard library XML module: `xml.etree.ElementTree`.
    """

    def __init__(self, *args, **kwargs) -> None: ...
    def parse_data(self) -> list[dict[str, str | None]]: ...

class _LxmlFrameParser(_XMLFrameParser):
    """
    Internal class to parse XML into DataFrames with third-party
    full-featured XML library, `lxml`, that supports
    XPath 1.0 and XSLT 1.0.
    """

    def __init__(self, *args, **kwargs) -> None: ...
    def parse_data(self) -> list[dict[str, str | None]]:
        """
        Parse xml data.

        This method will call the other internal methods to
        validate xpath, names, optionally parse and run XSLT,
        and parse original or transformed XML and return specific nodes.
        """
        ...

def get_data_from_filepath(
    filepath_or_buffer, encoding, compression, storage_options
) -> str | bytes | Buffer:
    """
    Extract raw XML data.

    The method accepts three input types:
        1. filepath (string-like)
        2. file-like object (e.g. open file object, StringIO)
        3. XML string or bytes

    This method turns (1) into (2) to simplify the rest of the processing.
    It returns input types (2) and (3) unchanged.
    """
    ...

def preprocess_data(data) -> io.StringIO | io.BytesIO:
    """
    Convert extracted raw data.

    This method will return underlying data of extracted XML content.
    The data either has a `read` attribute (e.g. a file object or a
    StringIO/BytesIO) or is a string or bytes that is an XML document.
    """
    ...

@doc(storage_options=_shared_docs["storage_options"])
def read_xml(
    path_or_buffer: FilePathOrBuffer,
    xpath: str | None = ...,
    namespaces: dict | list[dict] | None = ...,
    elems_only: bool | None = ...,
    attrs_only: bool | None = ...,
    names: list[str] | None = ...,
    encoding: str | None = ...,
    parser: str | None = ...,
    stylesheet: FilePathOrBuffer | None = ...,
    compression: CompressionOptions = ...,
    storage_options: StorageOptions = ...,
) -> DataFrame:
    r"""
    Read XML document into a ``DataFrame`` object.

    .. versionadded:: 1.3.0

    Parameters
    ----------
    path_or_buffer : str, path object, or file-like object
        Any valid XML string or path is acceptable. The string could be a URL.
        Valid URL schemes include http, ftp, s3, and file.

    xpath : str, optional, default './\*'
        The XPath to parse required set of nodes for migration to DataFrame.
        XPath should return a collection of elements and not a single
        element. Note: The ``etree`` parser supports limited XPath
        expressions. For more complex XPath, use ``lxml`` which requires
        installation.

    namespaces : dict, optional
        The namespaces defined in XML document as dicts with key being
        namespace prefix and value the URI. There is no need to include all
        namespaces in XML, only the ones used in ``xpath`` expression.
        Note: if XML document uses default namespace denoted as
        `xmlns='<URI>'` without a prefix, you must assign any temporary
        namespace prefix such as 'doc' to the URI in order to parse
        underlying nodes and/or attributes. For example, ::

            namespaces = {{"doc": "https://example.com"}}

    elems_only : bool, optional, default False
        Parse only the child elements at the specified ``xpath``. By default,
        all child elements and non-empty text nodes are returned.

    attrs_only :  bool, optional, default False
        Parse only the attributes at the specified ``xpath``.
        By default, all attributes are returned.

    names :  list-like, optional
        Column names for DataFrame of parsed XML data. Use this parameter to
        rename original element names and distinguish same named elements.

    encoding : str, optional, default 'utf-8'
        Encoding of XML document.

    parser : {{'lxml','etree'}}, default 'lxml'
        Parser module to use for retrieval of data. Only 'lxml' and
        'etree' are supported. With 'lxml' more complex XPath searches
        and ability to use XSLT stylesheet are supported.

    stylesheet : str, path object or file-like object
        A URL, file-like object, or a raw string containing an XSLT script.
        This stylesheet should flatten complex, deeply nested XML documents
        for easier parsing. To use this feature you must have ``lxml`` module
        installed and specify 'lxml' as ``parser``. The ``xpath`` must
        reference nodes of transformed XML document generated after XSLT
        transformation and not the original XML document. Only XSLT 1.0
        scripts and not later versions is currently supported.

    compression : {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}, default 'infer'
        For on-the-fly decompression of on-disk data. If 'infer', then use
        gzip, bz2, zip or xz if path_or_buffer is a string ending in
        '.gz', '.bz2', '.zip', or 'xz', respectively, and no decompression
        otherwise. If using 'zip', the ZIP file must contain only one data
        file to be read in. Set to None for no decompression.

    {storage_options}

    Returns
    -------
    df
        A DataFrame.

    See Also
    --------
    read_json : Convert a JSON string to pandas object.
    read_html : Read HTML tables into a list of DataFrame objects.

    Notes
    -----
    This method is best designed to import shallow XML documents in
    following format which is the ideal fit for the two-dimensions of a
    ``DataFrame`` (row by column). ::

            <root>
                <row>
                  <column1>data</column1>
                  <column2>data</column2>
                  <column3>data</column3>
                  ...
               </row>
               <row>
                  ...
               </row>
               ...
            </root>

    As a file format, XML documents can be designed any way including
    layout of elements and attributes as long as it conforms to W3C
    specifications. Therefore, this method is a convenience handler for
    a specific flatter design and not all possible XML structures.

    However, for more complex XML documents, ``stylesheet`` allows you to
    temporarily redesign original document with XSLT (a special purpose
    language) for a flatter version for migration to a DataFrame.

    This function will *always* return a single :class:`DataFrame` or raise
    exceptions due to issues with XML document, ``xpath``, or other
    parameters.

    Examples
    --------
    >>> xml = '''<?xml version='1.0' encoding='utf-8'?>
    ... <data xmlns="http://example.com">
    ...  <row>
    ...    <shape>square</shape>
    ...    <degrees>360</degrees>
    ...    <sides>4.0</sides>
    ...  </row>
    ...  <row>
    ...    <shape>circle</shape>
    ...    <degrees>360</degrees>
    ...    <sides/>
    ...  </row>
    ...  <row>
    ...    <shape>triangle</shape>
    ...    <degrees>180</degrees>
    ...    <sides>3.0</sides>
    ...  </row>
    ... </data>'''

    >>> df = pd.read_xml(xml)
    >>> df
          shape  degrees  sides
    0    square      360    4.0
    1    circle      360    NaN
    2  triangle      180    3.0

    >>> xml = '''<?xml version='1.0' encoding='utf-8'?>
    ... <data>
    ...   <row shape="square" degrees="360" sides="4.0"/>
    ...   <row shape="circle" degrees="360"/>
    ...   <row shape="triangle" degrees="180" sides="3.0"/>
    ... </data>'''

    >>> df = pd.read_xml(xml, xpath=".//row")
    >>> df
          shape  degrees  sides
    0    square      360    4.0
    1    circle      360    NaN
    2  triangle      180    3.0

    >>> xml = '''<?xml version='1.0' encoding='utf-8'?>
    ... <doc:data xmlns:doc="https://example.com">
    ...   <doc:row>
    ...     <doc:shape>square</doc:shape>
    ...     <doc:degrees>360</doc:degrees>
    ...     <doc:sides>4.0</doc:sides>
    ...   </doc:row>
    ...   <doc:row>
    ...     <doc:shape>circle</doc:shape>
    ...     <doc:degrees>360</doc:degrees>
    ...     <doc:sides/>
    ...   </doc:row>
    ...   <doc:row>
    ...     <doc:shape>triangle</doc:shape>
    ...     <doc:degrees>180</doc:degrees>
    ...     <doc:sides>3.0</doc:sides>
    ...   </doc:row>
    ... </doc:data>'''

    >>> df = pd.read_xml(xml,
    ...                  xpath="//doc:row",
    ...                  namespaces={{"doc": "https://example.com"}})
    >>> df
          shape  degrees  sides
    0    square      360    4.0
    1    circle      360    NaN
    2  triangle      180    3.0
    """
    ...
