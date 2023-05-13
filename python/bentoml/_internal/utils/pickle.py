from __future__ import annotations

import io
import pickle
import typing as t

# Pickle protocol 5 with out-of-band data. ref: https://peps.python.org/pep-0574/

# This is originally intended for numpy ndarray/pandas dataframe
# serialization.  In these situations the `main_bytes` part will only
# contain some metadata. That's why putting these bytes in header will
# not cause trouble. In the meantime the `concat_buffer_bytes`
# contains out-of-band buffers that need no computation during
# deserialization, which will save computation resource especially
# when the payload is large. However DO NOT use this pair of functions
# on `DefaultContainer`. Because `DefaultContainer` may be a
# dictionary containing a large object that has no out-of-band buffer
# (e.g. PIL Image or PyTorch tensor) and a small numpy ndarray. In
# that case `concat_buffer_bytes` will be small and `main_bytes` may
# be huge, hence we barely save any time while having a large header
# (that may cause errors).


def pep574_dumps(obj: t.Any) -> tuple[bytes, bytes, list[int]]:
    buffers: list[pickle.PickleBuffer] = []
    main_bytes: bytes = pickle.dumps(obj, protocol=5, buffer_callback=buffers.append)

    if not buffers:
        return main_bytes, b"", []

    buffer_bytess: list[bytes] = [buff.raw().tobytes() for buff in buffers]

    for buff in buffers:
        buff.release()

    indices: list[int] = [0]
    for buff_bytes in buffer_bytess:
        start = indices[-1]
        end = start + len(buff_bytes)
        indices.append(end)

    concat_buffer_bytes: bytes = b"".join(buffer_bytess)
    return main_bytes, concat_buffer_bytes, indices


def pep574_loads(
    main_bytes: bytes, concat_buffer_bytes: bytes, indices: list[int]
) -> t.Any:
    if not indices:
        # TODO: @larme monitor
        # https://github.com/pytorch/pytorch/issues/102977 and may change
        # this function later
        return fixed_torch_loads(main_bytes)

    mem = memoryview(concat_buffer_bytes)
    partitions = zip(indices, indices[1:])
    recover_buffers: list[pickle.PickleBuffer] = []
    for partition in partitions:
        buff = pickle.PickleBuffer(mem[slice(*partition)])
        recover_buffers.append(buff)

    return fixed_torch_loads(main_bytes, buffers=recover_buffers)


def _safe_torch_tensor_loads(bs: bytes) -> t.Any:
    import torch

    f = io.BytesIO(bs)
    if not torch.cuda.is_available():
        return torch.load(f, map_location="cpu")
    else:
        return torch.load(f)


class FixTorchUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> t.Callable[[bytes], t.Any]:
        if module == "torch.storage" and name == "_load_from_bytes":
            return _safe_torch_tensor_loads
        else:
            return super().find_class(module, name)


def fixed_torch_loads(bs: bytes, **kwargs: t.Any) -> t.Any:
    f = io.BytesIO(bs)
    unpickler = FixTorchUnpickler(f, **kwargs)
    return unpickler.load()


def loads_or_fix_torch(bs: bytes):
    try:
        return pickle.loads(bs)
    except RuntimeError:
        return fixed_torch_loads(bs)
