import typing as t

import numpy as np
from numpy.typing import DTypeLike
from numpy.typing import NDArray

def raise_error(msg: str) -> t.NoReturn:
    """
    Raise error with the provided message
    """
    ...

def serialized_byte_size(tensor_value: NDArray[t.Any]) -> int:
    """
    Get the underlying number of bytes for a numpy ndarray.

    Parameters
    ----------
    tensor_value : numpy.ndarray
        Numpy array to calculate the number of bytes for.

    Returns
    -------
    int
        Number of bytes present in this tensor
    """
    ...

class InferenceServerException(Exception):
    """Exception indicating non-Success status.

    Parameters
    ----------
    msg : str
        A brief description of error

    status : str
        The error code

    debug_details : str
        The additional details on the error

    """

    def __init__(
        self, msg: str, status: str = ..., debug_details: str = ...
    ) -> None: ...
    def __str__(self) -> str: ...
    def message(self) -> str:
        """Get the exception message.

        Returns
        -------
        str
            The message associated with this exception, or None if no message.

        """
        ...

    def status(self) -> str:
        """Get the status of the exception.

        Returns
        -------
        str
            Returns the status of the exception

        """
        ...

    def debug_details(self) -> str:
        """Get the detailed information about the exception
        for debugging purposes

        Returns
        -------
        str
            Returns the exception details

        """
        ...

def np_to_triton_dtype(np_dtype: DTypeLike) -> str: ...
def triton_to_np_dtype(dtype: str) -> DTypeLike: ...
def serialize_byte_tensor(input_tensor: NDArray[t.Any]) -> NDArray[np.uint8]:
    """
    Serializes a bytes tensor into a flat numpy array of length prepended
    bytes. The numpy array should use dtype of np.object_. For np.bytes_,
    numpy will remove trailing zeros at the end of byte sequence and because
    of this it should be avoided.

    Parameters
    ----------
    input_tensor : np.array
        The bytes tensor to serialize.

    Returns
    -------
    serialized_bytes_tensor : np.array
        The 1-D numpy array of type uint8 containing the serialized bytes in row-major form.

    Raises
    ------
    InferenceServerException
        If unable to serialize the given tensor.
    """
    ...

def deserialize_bytes_tensor(encoded_tensor: bytes) -> NDArray[t.Any]:
    """
    Deserializes an encoded bytes tensor into a
    numpy array of dtype of python objects

    Parameters
    ----------
    encoded_tensor : bytes
        The encoded bytes tensor where each element
        has its length in first 4 bytes followed by
        the content
    Returns
    -------
    string_tensor : np.array
        The 1-D numpy array of type object containing the
        deserialized bytes in row-major form.

    """
    ...

def serialize_bf16_tensor(input_tensor: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Serializes a bfloat16 tensor into a flat numpy array of bytes.
    The numpy array should use dtype of np.float32.

    Parameters
    ----------
    input_tensor : np.array
        The bfloat16 tensor to serialize.

    Returns
    -------
    serialized_bf16_tensor : np.array
        The 1-D numpy array of type uint8 containing the serialized bytes in row-major form.

    Raises
    ------
    InferenceServerException
        If unable to serialize the given tensor.
    """
    ...

def deserialize_bf16_tensor(encoded_tensor: bytes) -> NDArray[np.float32]:
    """
    Deserializes an encoded bf16 tensor into a
    numpy array of dtype of python objects

    Parameters
    ----------
    encoded_tensor : bytes
        The encoded bytes tensor where each element
        is 2 bytes (size of bfloat16)
    Returns
    -------
    string_tensor : np.array
        The 1-D numpy array of type float32 containing the
        deserialized bytes in row-major form.

    """
    ...
