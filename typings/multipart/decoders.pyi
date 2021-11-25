

import typing as t

class Base64Decoder:
    """This object provides an interface to decode a stream of Base64 data.  It
    is instantiated with an "underlying object", and whenever a write()
    operation is performed, it will decode the incoming data as Base64, and
    call write() on the underlying object.  This is primarily used for decoding
    form data encoded as Base64, but can be used for other purposes::

        from multipart.decoders import Base64Decoder
        fd = open("notb64.txt", "wb")
        decoder = Base64Decoder(fd)
        try:
            decoder.write("Zm9vYmFy")       # "foobar" in Base64
            decoder.finalize()
        finally:
            decoder.close()

        # The contents of "notb64.txt" should be "foobar".

    This object will also pass all finalize() and close() calls to the
    underlying object, if the underlying object supports them.

    Note that this class maintains a cache of base64 chunks, so that a write of
    arbitrary size can be performed.  You must call :meth:`finalize` on this
    object after all writes are completed to ensure that all data is flushed
    to the underlying object.

    :param underlying: the underlying object to pass writes to
    """

    def __init__(self, underlying: t.BinaryIO) -> None: ...
    def write(self, data: t.AnyStr) -> None:
        """Takes any input data provided, decodes it as base64, and passes it
        on to the underlying object.  If the data provided is invalid base64
        data, then this method will raise
        a :class:`multipart.exceptions.DecodeError`

        :param data: base64 data to decode
        """
        ...
    def close(self) -> None:
        """Close this decoder.  If the underlying object has a `close()`
        method, this function will call it.
        """
        ...
    def finalize(self) -> None:
        """Finalize this object.  This should be called when no more data
        should be written to the stream.  This function can raise a
        :class:`multipart.exceptions.DecodeError` if there is some remaining
        data in the cache.

        If the underlying object has a `finalize()` method, this function will
        call it.
        """
        ...
    def __repr__(self) -> str: ...

class QuotedPrintableDecoder:
    """This object provides an interface to decode a stream of quoted-printable
    data.  It is instantiated with an "underlying object", in the same manner
    as the :class:`multipart.decoders.Base64Decoder` class.  This class behaves
    in exactly the same way, including maintaining a cache of quoted-printable
    chunks.

    :param underlying: the underlying object to pass writes to
    """

    def __init__(self, underlying: t.BinaryIO) -> None: ...
    def write(self, data: t.AnyStr) -> None:
        """Takes any input data provided, decodes it as quoted-printable, and
        passes it on to the underlying object.

        :param data: quoted-printable data to decode
        """
        ...
    def close(self) -> None:
        """Close this decoder.  If the underlying object has a `close()`
        method, this function will call it.
        """
        ...
    def finalize(self) -> None:
        """Finalize this object.  This should be called when no more data
        should be written to the stream.  This function will not raise any
        exceptions, but it may write more data to the underlying object if
        there is data remaining in the cache.

        If the underlying object has a `finalize()` method, this function will
        call it.
        """
        ...
    def __repr__(self) -> str: ...
