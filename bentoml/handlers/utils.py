TF_B64_KEY = 'b64'


def tf_b64_2_bytes(obj):
    import base64

    if isinstance(obj, dict) and TF_B64_KEY in obj:
        return base64.b64decode(obj[TF_B64_KEY])
    else:
        return obj


def bytes_2_tf_b64(obj):
    import base64

    if isinstance(obj, bytes):
        return {TF_B64_KEY: base64.b64encode(obj).decode('utf-8')}
    else:
        return obj


def tf_tensor_2_serializable(obj):
    '''
    To convert
        tf.Tensor -> json serializable
        np.ndarray -> json serializable
        bytes -> {'b64': <b64_str>}
        others -> themselves
    '''
    import tensorflow as tf
    import numpy as np

    # Tensor -> ndarray or object
    if isinstance(obj, tf.Tensor):
        if tf.__version__.startswith("1."):
            with tf.compat.v1.Session():
                obj = obj.numpy()
        else:
            obj = obj.numpy()

    # ndarray -> serializable python object
    TYPES = (int, float, str)
    if isinstance(obj, np.ndarray):
        for _type in TYPES:
            # dtype of string/bytes ndarrays returned by tensor.numpy()
            # are both np.dtype(object), which are not json serializable
            try:
                obj = obj.astype(_type)
            except (UnicodeDecodeError, ValueError, OverflowError):
                continue
            break
        else:
            obj = np.vectorize(bytes_2_tf_b64)(obj)
        obj = obj.tolist()
    elif isinstance(obj, bytes):
        # tensor.numpy() will return single value directly
        try:
            obj = obj.decode("utf8")
        except UnicodeDecodeError:
            obj = bytes_2_tf_b64(obj)

    return obj


class NestedConverter:
    '''
    Generate a nested converter that supports object in list/tuple/dict
    from a single converter.
    '''

    def __init__(self, converter):
        self.converter = converter

    def __call__(self, obj):
        converted = self.converter(obj)
        if obj is obj and converted is not obj:
            return converted

        if isinstance(obj, dict):
            return {k: self(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self(v) for v in obj]
        else:
            return obj
