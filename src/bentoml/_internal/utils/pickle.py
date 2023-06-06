from __future__ import annotations

import io
import sys
import typing as t

if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


# Pickle protocol 5 with out-of-band data
# https://peps.python.org/pep-0574/
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
        # TODO: @larme monitor https://github.com/pytorch/pytorch/issues/102977 and may use this function later
        return _fix_torch_loads(main_bytes)

    mem = memoryview(concat_buffer_bytes)
    partitions = zip(indices, indices[1:])
    recover_buffers: list[pickle.PickleBuffer] = []
    for partition in partitions:
        buff = pickle.PickleBuffer(mem[slice(*partition)])
        recover_buffers.append(buff)

    return pickle.loads(main_bytes, buffers=recover_buffers)


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


def _fix_torch_loads(bs: bytes) -> t.Any:
    f = io.BytesIO(bs)
    unpickler = FixTorchUnpickler(f)
    return unpickler.load()


def loads_or_fix_torch(bs: bytes):
    try:
        return pickle.loads(bs)
    except RuntimeError:
        return _fix_torch_loads(bs)
