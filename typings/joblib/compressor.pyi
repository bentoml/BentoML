import io

"""Classes and functions for managing compressors."""
LZ4_NOT_INSTALLED_ERROR = ...
_COMPRESSORS = ...
_ZFILE_PREFIX = ...
_ZLIB_PREFIX = ...
_GZIP_PREFIX = ...
_BZ2_PREFIX = ...
_XZ_PREFIX = ...
_LZMA_PREFIX = ...
_LZ4_PREFIX = ...

def register_compressor(compressor_name, compressor, force=...):  # -> None:
    """Register a new compressor.

    Parameters
    -----------
    compressor_name: str.
        The name of the compressor.
    compressor: CompressorWrapper
        An instance of a 'CompressorWrapper'.
    """
    ...

class CompressorWrapper:
    """A wrapper around a compressor file object.

    Attributes
    ----------
    obj: a file-like object
        The object must implement the buffer interface and will be used
        internally to compress/decompress the data.
    prefix: bytestring
        A bytestring corresponding to the magic number that identifies the
        file format associated to the compressor.
    extention: str
        The file extension used to automatically select this compressor during
        a dump to a file.
    """

    def __init__(self, obj, prefix=..., extension=...) -> None: ...
    def compressor_file(self, fileobj, compresslevel=...):
        """Returns an instance of a compressor file object."""
        ...
    def decompressor_file(self, fileobj):
        """Returns an instance of a decompressor file object."""
        ...

class BZ2CompressorWrapper(CompressorWrapper):
    prefix = ...
    extension = ...
    def __init__(self) -> None: ...
    def compressor_file(self, fileobj, compresslevel=...):  # -> BZ2File:
        """Returns an instance of a compressor file object."""
        ...
    def decompressor_file(self, fileobj):  # -> BZ2File:
        """Returns an instance of a decompressor file object."""
        ...

class LZMACompressorWrapper(CompressorWrapper):
    prefix = ...
    extension = ...
    _lzma_format_name = ...
    def __init__(self) -> None: ...
    def compressor_file(self, fileobj, compresslevel=...):  # -> LZMAFile:
        """Returns an instance of a compressor file object."""
        ...
    def decompressor_file(self, fileobj):  # -> LZMAFile:
        """Returns an instance of a decompressor file object."""
        ...

class XZCompressorWrapper(LZMACompressorWrapper):
    prefix = ...
    extension = ...
    _lzma_format_name = ...

class LZ4CompressorWrapper(CompressorWrapper):
    prefix = ...
    extension = ...
    def __init__(self) -> None: ...
    def compressor_file(self, fileobj, compresslevel=...):
        """Returns an instance of a compressor file object."""
        ...
    def decompressor_file(self, fileobj):
        """Returns an instance of a decompressor file object."""
        ...

_MODE_CLOSED = ...
_MODE_READ = ...
_MODE_READ_EOF = ...
_MODE_WRITE = ...
_BUFFER_SIZE = ...

class BinaryZlibFile(io.BufferedIOBase):
    """A file object providing transparent zlib (de)compression.

    TODO python2_drop: is it still needed since we dropped Python 2 support A
    BinaryZlibFile can act as a wrapper for an existing file object, or refer
    directly to a named file on disk.

    Note that BinaryZlibFile provides only a *binary* file interface: data read
    is returned as bytes, and data to be written should be given as bytes.

    This object is an adaptation of the BZ2File object and is compatible with
    versions of python >= 2.7.

    If filename is a str or bytes object, it gives the name
    of the file to be opened. Otherwise, it should be a file object,
    which will be used to read or write the compressed data.

    mode can be 'rb' for reading (default) or 'wb' for (over)writing

    If mode is 'wb', compresslevel can be a number between 1
    and 9 specifying the level of compression: 1 produces the least
    compression, and 9 produces the most compression. 3 is the default.
    """

    wbits = ...
    def __init__(self, filename, mode=..., compresslevel=...) -> None: ...
    def close(self):  # -> None:
        """Flush and close the file.

        May be called more than once without error. Once the file is
        closed, any other operation on it will raise a ValueError.
        """
        ...
    @property
    def closed(self):  # -> bool:
        """True if this file is closed."""
        ...
    def fileno(self):  # -> int:
        """Return the file descriptor for the underlying file."""
        ...
    def seekable(self):  # -> bool:
        """Return whether the file supports seeking."""
        ...
    def readable(self):  # -> bool:
        """Return whether the file was opened for reading."""
        ...
    def writable(self):  # -> bool:
        """Return whether the file was opened for writing."""
        ...
    def read(self, size=...):  # -> bytes | None:
        """Read up to size uncompressed bytes from the file.

        If size is negative or omitted, read until EOF is reached.
        Returns b'' if the file is already at EOF.
        """
        ...
    def readinto(self, b):  # -> int:
        """Read up to len(b) bytes into b.

        Returns the number of bytes read (0 for EOF).
        """
        ...
    def write(self, data):  # -> int:
        """Write a byte string to the file.

        Returns the number of uncompressed bytes written, which is
        always len(data). Note that due to buffering, the file on disk
        may not reflect the data written until close() is called.
        """
        ...
    def seek(self, offset, whence=...):  # -> int:
        """Change the file position.

        The new position is specified by offset, relative to the
        position indicated by whence. Values for whence are:

            0: start of stream (default); offset must not be negative
            1: current stream position
            2: end of stream; offset must not be positive

        Returns the new file position.

        Note that seeking is emulated, so depending on the parameters,
        this operation may be extremely slow.
        """
        ...
    def tell(self):  # -> int:
        """Return the current file position."""
        ...

class ZlibCompressorWrapper(CompressorWrapper):
    def __init__(self) -> None: ...

class BinaryGzipFile(BinaryZlibFile):
    """A file object providing transparent gzip (de)compression.

    If filename is a str or bytes object, it gives the name
    of the file to be opened. Otherwise, it should be a file object,
    which will be used to read or write the compressed data.

    mode can be 'rb' for reading (default) or 'wb' for (over)writing

    If mode is 'wb', compresslevel can be a number between 1
    and 9 specifying the level of compression: 1 produces the least
    compression, and 9 produces the most compression. 3 is the default.
    """

    wbits = ...

class GzipCompressorWrapper(CompressorWrapper):
    def __init__(self) -> None: ...
