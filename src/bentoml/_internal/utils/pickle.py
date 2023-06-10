from __future__ import annotations

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
        return pickle.loads(main_bytes)

    mem = memoryview(concat_buffer_bytes)
    partitions = zip(indices, indices[1:])
    recover_buffers: list[pickle.PickleBuffer] = []
    for partition in partitions:
        buff = pickle.PickleBuffer(mem[slice(*partition)])
        recover_buffers.append(buff)

    return pickle.loads(main_bytes, buffers=recover_buffers)
